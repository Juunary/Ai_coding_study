import os
import argparse
import numpy as np
import open3d as o3d
import torch


def analyze_ply_gpu(path_in: str, downsample_N: int = 20000, output_xyz: str = None):
    """
    PLY 또는 OBJ 파일을 읽고 GPU로 가속하여:
      • 최대 축 길이 (mm)
      • 표면적 (mm²)
      • 부피 (mm³)
    를 계산하고, XYZ 포인트 클라우드를 저장합니다.
    """
    # 1) 메쉬 로드 및 포인트 샘플링
    mesh = o3d.io.read_triangle_mesh(path_in)
    if mesh.has_triangles():
        pcd = mesh.sample_points_poisson_disk(number_of_points=downsample_N)
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
    if pts.shape[0] > downsample_N:
        idx = torch.randperm(pts.shape[0], device="cuda")[:downsample_N]
        pts = pts[idx]
        pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())

    # 5) Surface Area & Volume 계산
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
    parser.add_argument("-n", "--downsample", type=int, default=10000, help="포인트 샘플링 수 (기본: 50000)")
    args = parser.parse_args()

    analyze_ply_gpu(args.input_ply, args.downsample, args.output_xyz)
