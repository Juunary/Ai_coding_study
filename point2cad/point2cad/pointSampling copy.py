#!/usr/bin/env python3
import os
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path


def convert_to_xyz(path_in: str, downsample_N: int = 20000):
    """
    단일 PLY 파일 → 점군으로 변환 후 .xyz 저장
    """
    mesh = o3d.io.read_triangle_mesh(path_in)
    if mesh.has_triangles():
        pcd = mesh.sample_points_poisson_disk(number_of_points=downsample_N)
    else:
        pcd = o3d.io.read_point_cloud(path_in)
        if len(pcd.points) > downsample_N:
            pts = np.asarray(pcd.points)
            idx = np.random.choice(len(pts), downsample_N, replace=False)
            pcd.points = o3d.utility.Vector3dVector(pts[idx])

    out_path = Path(path_in).with_name(Path(path_in).stem + "_point.xyz")
    np.savetxt(out_path, np.asarray(pcd.points), fmt="%.6f")
    print(f"[OK] {path_in} → {out_path} (N={len(pcd.points)})")


def main():
    parser = argparse.ArgumentParser(description="폴더 내 모든 PLY 파일을 점군 .xyz로 변환")
    parser.add_argument("--path_in", required=True, help="입력 PLY 파일들이 들어있는 폴더 경로")
    parser.add_argument("-n", "--downsample", type=int, default=20000, help="포인트 수 (기본: 20000)")
    args = parser.parse_args()

    folder = Path(args.path_in)
    if not folder.is_dir():
        print(f"[ERR] 폴더 경로가 아닙니다: {folder}")
        return

    ply_files = list(folder.glob("*.ply"))
    if not ply_files:
        print(f"[INFO] {folder} 안에 .ply 파일이 없습니다.")
        return

    for ply in ply_files:
        convert_to_xyz(str(ply), downsample_N=args.downsample)

    print(f"[DONE] 총 {len(ply_files)} 개 파일 변환 완료.")


if __name__ == "__main__":
    main()
