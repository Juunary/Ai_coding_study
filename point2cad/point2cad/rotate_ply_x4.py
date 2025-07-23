#!/usr/bin/env python3
import argparse
import trimesh
import numpy as np
import os

def generate_transformed_meshes(mesh, axis_point, radius, copies=4):
    meshes = []
    for i in range(copies):
        angle = 2 * np.pi * i / copies
        offset = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            0  # z축 방향으로는 위치 고정
        ])
        target_point = axis_point + offset
        translation = target_point - mesh.centroid
        new_mesh = mesh.copy()
        new_mesh.apply_translation(translation)
        meshes.append(new_mesh)
    return trimesh.util.concatenate(meshes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="z축 병행 중심축 기준 복제 저장")
    parser.add_argument("--path_in", type=str, required=True, help=".ply 입력 경로")
    parser.add_argument("--radius", type=float, default=0.0, help="회전 복제 반지름")
    parser.add_argument("--x", type=float, default=0.0, help="중심축 위치 x좌표")
    parser.add_argument("--y", type=float, default=0.0, help="중심축 위치 y좌표")
    args = parser.parse_args()

    path_in = args.path_in
    if not os.path.exists(path_in):
        raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {path_in}")

    mesh = trimesh.load(path_in)
    axis_point = np.array([args.x, args.y, mesh.centroid[2]])  # z는 무시하고 중심 높이 유지

    duplicated = generate_transformed_meshes(mesh, axis_point, args.radius)

    base, ext = os.path.splitext(path_in)
    path_out = base + "X4" + ext
    duplicated.export(path_out)
    print(f"복제 메쉬 저장 완료: {path_out}")
