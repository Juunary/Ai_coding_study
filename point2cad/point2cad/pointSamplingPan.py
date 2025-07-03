import os
import argparse
import numpy as np
import open3d as o3d
import torch

def extract_thickness_faces(mesh, thickness_threshold=0.1):
    # 모든 triangle의 normal 계산
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    v0, v1, v2 = vertices[triangles[:,0]], vertices[triangles[:,1]], vertices[triangles[:,2]]
    normals = np.cross(v1-v0, v2-v0)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    # 두께(엣지): 법선이 Z축과 거의 수직 (X/Y축 방향), |nz| < 임계값
    is_thickness = np.abs(normals[:,2]) < thickness_threshold
    return np.where(is_thickness)[0]

def sample_more_on_thickness(mesh, n_samples=20000, edge_multiplier=5, thickness_threshold=0.15):
    # 전체 Poisson 샘플링(표면 위주)
    base_pts = np.asarray(mesh.sample_points_poisson_disk(n_samples).points)
    # 두께면(엣지) triangle 추출
    edge_face_idx = extract_thickness_faces(mesh, thickness_threshold)
    edge_triangles = np.asarray(mesh.triangles)[edge_face_idx]
    edge_verts = np.asarray(mesh.vertices)
    # 각 엣지 삼각형에서 uniform random 샘플링
    samples = []
    for tri in edge_triangles:
        v0, v1, v2 = edge_verts[tri]
        for _ in range(edge_multiplier):
            a, b = np.random.rand(2)
            if a + b > 1:
                a, b = 1-a, 1-b
            sample = v0 + a*(v1-v0) + b*(v2-v0)
            samples.append(sample)
    edge_pts = np.array(samples) if len(samples) > 0 else np.empty((0,3))
    # 두 샘플 합치기
    total_pts = np.vstack([base_pts, edge_pts])
    return total_pts

def analyze_ply_gpu(path_in: str, downsample_N: int = 20000, output_xyz: str = None, edge_multiplier: int = 5):
    mesh = o3d.io.read_triangle_mesh(path_in)
    if mesh.has_triangles():
        pts_np = sample_more_on_thickness(mesh, n_samples=downsample_N, edge_multiplier=edge_multiplier)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_np)
    else:
        pcd = o3d.io.read_point_cloud(path_in)

    # 2) GPU Tensor로 변환
    pts = torch.from_numpy(np.asarray(pcd.points)).float().cuda()

    # 3) AABB 계산
    min_vals, _ = pts.min(dim=0)
    max_vals, _ = pts.max(dim=0)
    extents = (max_vals - min_vals).cpu().numpy()
    print(f"■ 최대 길이 (GPU): {np.round(extents.max(), 6)} mm")

    # 4) 과샘플링 시 다운샘플링
    if pts.shape[0] > downsample_N*2:
        idx = torch.randperm(pts.shape[0], device="cuda")[:downsample_N*2]
        pts = pts[idx]
        pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())

    # 5) Surface Area & Volume 계산 (기존과 동일)
    verts = torch.from_numpy(np.asarray(mesh.vertices)).float().cuda()
    tris = torch.from_numpy(np.asarray(mesh.triangles)).long().cuda()
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    cross_prod = torch.cross(v1 - v0, v2 - v0, dim=1)
    areas = torch.norm(cross_prod, dim=1) * 0.5
    total_area = areas.sum().item()
    print(f"■ GPU 계산 표면적: {total_area:.6f} (단위: mm²)")

    signed_vols = torch.sum(torch.sum(torch.cross(v0, v1, dim=1) * v2, dim=1)) / 6.0
    total_volume = abs(signed_vols.item())
    print(f"■ GPU 계산 부피(volume): {total_volume:.6f} (단위: mm³)")

    # 6) XYZ 저장
    if output_xyz is None:
        base = os.path.splitext(os.path.basename(path_in))[0]
        out_dir = os.path.join("..", "assets", "xyz")
        os.makedirs(out_dir, exist_ok=True)
        output_xyz = os.path.join(out_dir, f"{base}_gpu.xyz")

    np.savetxt(output_xyz, np.asarray(pcd.points), fmt="%.6f")
    print(f"→ 포인트 클라우드를 '{output_xyz}'에 저장했습니다.")

    return pcd.points, total_area, total_volume

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLY 파일 분석 & XYZ 변환 (GPU 가속, mm 단위, 엣지 우선 샘플링)")
    parser.add_argument("input_ply", help="입력 PLY 또는 OBJ 파일 경로")
    parser.add_argument("output_xyz", nargs="?", default=None, help="출력할 .xyz 파일 경로 (생략 시 자동)")
    parser.add_argument("-n", "--downsample", type=int, default=10000, help="포인트 샘플링 수 (기본: 10000)")
    parser.add_argument("--edge_multiplier", type=int, default=5, help="엣지(두께면)에서 샘플링 곱셈 계수 (기본: 5)")
    args = parser.parse_args()

    analyze_ply_gpu(args.input_ply, args.downsample, args.output_xyz, args.edge_multiplier)
