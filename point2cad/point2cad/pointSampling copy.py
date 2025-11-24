import os
import argparse
import numpy as np
import open3d as o3d
import torch
from collections import defaultdict


def sample_points_core_centered(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int,
    center_alpha: float = 5.0,
    drop_boundary_layer: bool = True,
) -> o3d.geometry.PointCloud:
    """
    삼각형 메쉬에서 점군을 샘플링하되,

      - 메쉬 외곽(boundary) 삼각형은 제외해서
        모서리/끝부분에는 점이 안 찍히고
      - 내부 삼각형의 중심 쪽으로 점이 몰리도록

    한다.

    center_alpha > 1:
        클수록 각 삼각형의 중앙(1/3,1/3,1/3) 근처에 점이 더 밀집.
    drop_boundary_layer:
        True 이면 바운더리에 인접한 삼각형(face)은 전부 버리고,
        내부(face)에서만 샘플링.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)

    if triangles.size == 0:
        raise ValueError("메쉬에 삼각형(face)이 없습니다. 점군 샘플링 불가.")

    # --- 1) boundary vertex 검출 ---
    edge_faces = defaultdict(list)  # (min(i,j), max(i,j)) -> [face_idx, ...]
    for fi, tri in enumerate(triangles):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        edges = [(a, b), (b, c), (c, a)]
        for v1, v2 in edges:
            key = (v1, v2) if v1 < v2 else (v2, v1)
            edge_faces[key].append(fi)

    n_verts = vertices.shape[0]
    boundary_vertex_flag = np.zeros(n_verts, dtype=np.bool_)

    for (v1_idx, v2_idx), faces in edge_faces.items():
        # boundary edge: 딱 하나의 face만 가지는 edge
        if len(faces) == 1:
            boundary_vertex_flag[v1_idx] = True
            boundary_vertex_flag[v2_idx] = True

    # face가 boundary에 붙어 있는지 여부: 세 버텍스 중 하나라도 boundary vertex면 True
    face_is_boundary = boundary_vertex_flag[triangles].any(axis=1)  # (F,)

    # --- 2) core(내부) triangle만 사용 (옵션) ---
    if drop_boundary_layer:
        core_mask = ~face_is_boundary
        if not np.any(core_mask):
            # 내부 face가 하나도 없으면, 그냥 전체를 사용
            print("[경고] 내부 삼각형이 없어 boundary layer를 제거하지 않습니다.")
            core_triangles = triangles
        else:
            core_triangles = triangles[core_mask]
    else:
        core_triangles = triangles

    # --- 3) core triangle들의 면적 기반 가중치 ---
    v0 = vertices[core_triangles[:, 0]]
    v1 = vertices[core_triangles[:, 1]]
    v2 = vertices[core_triangles[:, 2]]

    cross_prod = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross_prod, axis=1)  # (F_core,)

    # 면적이 0인 face는 제거
    valid_mask = areas > 0.0
    if not np.any(valid_mask):
        raise ValueError("모든 삼각형 면적이 0입니다. 메쉬 상태가 비정상적입니다.")

    core_triangles = core_triangles[valid_mask]
    areas = areas[valid_mask]
    v0 = v0[valid_mask]
    v1 = v1[valid_mask]
    v2 = v2[valid_mask]

    probs = areas / areas.sum()

    # --- 4) 면적 비례로 triangle 선택 ---
    tri_indices = np.random.choice(len(core_triangles), size=num_points, p=probs)
    tri_v0 = v0[tri_indices]
    tri_v1 = v1[tri_indices]
    tri_v2 = v2[tri_indices]

    # --- 5) 각 삼각형 내부에서 중앙에 치우친 바리센트릭 샘플링 ---
    alpha = float(center_alpha)
    if alpha <= 1.0:
        raise ValueError("center_alpha는 1보다 큰 값으로 설정해야 합니다.")

    # Dirichlet(alpha, alpha, alpha) ~ 중앙 쪽에 밀집
    bary = np.random.gamma(shape=alpha, scale=1.0, size=(num_points, 3)).astype(
        np.float32
    )
    bary /= bary.sum(axis=1, keepdims=True)  # (N,3)

    pts = (
        tri_v0 * bary[:, [0]]
        + tri_v1 * bary[:, [1]]
        + tri_v2 * bary[:, [2]]
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def analyze_ply_gpu(path_in: str, downsample_N: int = 20000, output_xyz: str = None):
    """
    PLY 또는 OBJ 파일을 읽고 GPU로 가속하여:
      • 최대 축 길이 (mm)   → 메쉬 전체 AABB 기준
      • 표면적 (mm²)        → 삼각형 메쉬 기준
      • 부피 (mm³)          → 삼각형 메쉬 기준
    를 계산하고, XYZ 포인트 클라우드를 저장합니다.

    포인트 클라우드는:
      - 메쉬 외곽(boundary)에 붙은 삼각형은 쓰지 않고
      - 내부 삼각형의 중앙 근처에만 점을 찍어서
        모서리/끝부분에는 점이 안 찍히도록 만듭니다.
    """
    # 1) 메쉬 로드
    mesh = o3d.io.read_triangle_mesh(path_in)

    # 2) AABB / 길이 계산은 "원래 메쉬" 기준으로 수행
    verts_np = np.asarray(mesh.vertices)
    if verts_np.size > 0:
        verts_t = torch.from_numpy(verts_np).float().cuda()
        min_vals, _ = verts_t.min(dim=0)
        max_vals, _ = verts_t.max(dim=0)
        extents = (max_vals - min_vals).cpu().numpy()
        print(f"■ 최대 길이 (GPU, 메쉬 기준): {np.round(extents.max(), 6)} mm")
    else:
        print("■ 경고: 메쉬에 버텍스가 없습니다. 최대 길이를 계산할 수 없습니다.")
        extents = np.zeros(3, dtype=np.float32)

    # 3) 포인트 샘플링 (메쉬가 삼각형을 갖는 경우에만)
    if mesh.has_triangles():
        pcd = sample_points_core_centered(
            mesh,
            num_points=downsample_N,
            center_alpha=5.0,        # 클수록 각 face 중앙에 더 몰림
            drop_boundary_layer=True # True이면 바운더리 한 겹을 통째로 제거
        )
    else:
        # 삼각형이 없으면 그냥 점군으로 읽고, 이후 로직만 동일 적용
        print("■ 메쉬에 삼각형이 없어 입력을 포인트 클라우드로 처리합니다.")
        pcd = o3d.io.read_point_cloud(path_in)

    pts_np = np.asarray(pcd.points)
    if pts_np.size == 0:
        raise ValueError("샘플링된 포인트가 없습니다.")

    # 4) 필요 시 다운샘플링 (현재는 정확히 num_points만 찍으므로 거의 사용 안됨)
    pts = torch.from_numpy(pts_np).float().cuda()
    if pts.shape[0] > downsample_N:
        idx = torch.randperm(pts.shape[0], device="cuda")[:downsample_N]
        pts = pts[idx]
        pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())

    # 5) Surface Area & Volume 계산 (메쉬 기준)
    if mesh.has_triangles():
        verts = torch.from_numpy(np.asarray(mesh.vertices)).float().cuda()
        tris_np = np.asarray(mesh.triangles, dtype=np.int64)
        if tris_np.size > 0:
            tris = torch.from_numpy(tris_np).long().cuda()
            v0 = verts[tris[:, 0]]
            v1 = verts[tris[:, 1]]
            v2 = verts[tris[:, 2]]

            cross_prod = torch.cross(v1 - v0, v2 - v0, dim=1)
            areas = torch.norm(cross_prod, dim=1) * 0.5
            total_area = areas.sum().item()
            print(f"■ GPU 계산 표면적: {total_area:.6f} (단위: mm²)")

            signed_vols = torch.sum(
                torch.sum(torch.cross(v0, v1, dim=1) * v2, dim=1)
            ) / 6.0
            total_volume = abs(signed_vols.item())
            print(f"■ GPU 계산 부피(volume): {total_volume:.6f} (단위: mm³)")
        else:
            total_area = 0.0
            total_volume = 0.0
            print("■ 경고: 삼각형(face)이 없어 표면적/부피 계산을 건너뜁니다.")
    else:
        total_area = 0.0
        total_volume = 0.0
        print("■ 경고: 메쉬에 삼각형(face)이 없어 표면적/부피 계산을 건너뜁니다.")

    # 6) XYZ 저장 경로 설정
    if output_xyz is None:
        base = os.path.splitext(os.path.basename(path_in))[0]
        out_dir = os.path.join("..", "assets", "xyz")
        os.makedirs(out_dir, exist_ok=True)
        output_xyz = os.path.join(out_dir, f"{base}_gpu.xyz")

    np.savetxt(output_xyz, np.asarray(pcd.points), fmt="%.6f")
    print(f"→ 포인트 클라우드를 '{output_xyz}'에 저장했습니다.")

    return pcd.points, total_area, total_volume


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLY 파일 분석 & XYZ 변환 (GPU 가속, mm 단위)")
    parser.add_argument("input_ply", help="입력 PLY 또는 OBJ 파일 경로")
    parser.add_argument("output_xyz", nargs="?", default=None, help="출력할 .xyz 파일 경로 (생략 시 자동)")
    parser.add_argument(
        "-n",
        "--downsample",
        type=int,
        default=10000,
        help="포인트 샘플링 수 (기본: 10000)",
    )
    args = parser.parse_args()

    analyze_ply_gpu(args.input_ply, args.downsample, args.output_xyz)
