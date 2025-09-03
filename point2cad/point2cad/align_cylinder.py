#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# usage: python align_cylinder.py --ply ../assets/ply/shaft/mesh_001.ply

import argparse
import numpy as np
import open3d as o3d

def main(ply_path):
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if mesh.is_empty():
        raise ValueError(f"메쉬가 비어있습니다: {ply_path}")
    pts = np.asarray(mesh.vertices)

    # 중심 정렬
    pts = pts - pts.mean(axis=0, keepdims=True)

    # PCA (공분산 행렬)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 가장 큰 고유값의 벡터 = 긴 축
    main_axis = eigvecs[:, np.argmax(eigvals)]
    main_axis /= np.linalg.norm(main_axis)

    # 기준 벡터: z축
    z_axis = np.array([1.0, 0, 0])

    # 두 벡터 사이의 각도 (라디안 → 도)
    cos_angle = np.clip(np.dot(main_axis, z_axis), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))

    # --- 회전축이 X축으로만 제한된 경우 ---
    # main_axis의 y,z 성분을 사용해서 atan2 계산
    yz_angle = np.degrees(np.arctan2(main_axis[1], main_axis[2]))
    # 이 각도를 빼줘야 z축에 정렬됨
    rot_x_deg = -yz_angle

    print("=== Cylinder Axis Alignment ===")
    print(f"Main axis vector (from PCA): {main_axis}")
    print(f"Angle to Z (any-axis rotation): {angle_deg:.4f} deg")
    print(f"→ Required rotation about X-axis: {rot_x_deg:.4f} deg")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ply", required=True, help="입력 PLY 파일 경로")
    args = p.parse_args()
    main(args.ply)
