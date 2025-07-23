#!/usr/bin/env python3
import argparse
import trimesh
import numpy as np
import os

def duplicate_around_axis_rotated(mesh, axis_point, radius, copies=4):
    meshes = []
    for i in range(copies):
        angle = 2 * np.pi * i / copies

        # 위치 계산 (xy 평면의 원 위)
        offset = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            0
        ])
        target_position = axis_point + offset
        translation = target_position - mesh.centroid

        # 회전 행렬: z축 기준 회전 (origin 기준)
        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle=angle,
            direction=[0, 0, 1],
            point=mesh.centroid  # 중심 기준 회전
        )

        new_mesh = mesh.copy()
        new_mesh.apply_transform(rotation_matrix)       # z축 회전
        new_mesh.apply_translation(translation)         # 위치 이동
        meshes.append(new_mesh)

    return trimesh.util.concatenate(meshes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="중심축 기준 복제 + z축 회전 저장")
    parser.add_argument("--path_in", type=str, required=True, help=".ply 입력 경로")
    parser.add_argument("--radius", type=float, default=0.0, help="복제 중심축에서 떨어진 거리")
    parser.add_argument("--x", type=float, default=0.0, help="중심축 기준점 x")
    parser.add_argument("--y", type=float, default=0.0, help="중심축 기준점 y")
    args = parser.parse_args()

    if not os.path.exists(args.path_in):
        raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {args.path_in}")

    mesh = trimesh.load(args.path_in)
    axis_point = np.array([args.x, args.y, mesh.centroid[2]])  # z 위치 유지

    duplicated = duplicate_around_axis_rotated(mesh, axis_point, args.radius)

    out_path = os.path.splitext(args.path_in)[0] + "X4.ply"
    duplicated.export(out_path)
    print(f"복제 완료: {out_path}")
