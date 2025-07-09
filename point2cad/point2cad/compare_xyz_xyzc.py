#!/usr/bin/env python3
# compare_xyz_vs_xyz.py
"""
원본 XYZ 와 XYZC(→XYZ) 의 기하학 정확도(표면적·부피) 비교
  1) 두 점군 로드
  2) XYZC 에서 c 컬럼 제거
  3) Similarity(Scale·R·T) ICP 로 좌표계 맞춤
  4) Poisson 재구성 → 면적·부피 계산
  5) 상대 오차 출력
Usage:
    python compare_xyz_xyz.py  --xyz  model.xyz   --xyzc segment.xyzc
"""

import argparse, sys, os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


# ──────────────────────────────────────────────────────────────────────────────
# Helper : PointCloud 3D txt 로드
def load_xyz_txt(path: str, drop_extra: bool = False) -> o3d.geometry.PointCloud:
    arr = np.loadtxt(path)
    if drop_extra and arr.shape[1] > 3:
        arr = arr[:, :3]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(arr.astype(np.float64)))
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound()) * 0.03,
                max_nn=30,
            )
        )
    return pcd


# Helper : 메쉬 → 면적·부피
def mesh_area_volume(mesh: o3d.geometry.TriangleMesh):
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    v0, v1, v2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]

    tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    area = tri_areas.sum()

    signed_vol = (np.einsum("ij,ij->i", np.cross(v0, v1), v2)).sum() / 6.0
    volume = abs(signed_vol)
    return area, volume


# Helper : Poisson 재구성(깊이 9) 후 자잘한 정리
def poisson_reconstruct(pcd: o3d.geometry.PointCloud, depth: int = 9):
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    return mesh


# ──────────────────────────────────────────────────────────────────────────────
def main(xyz_path: str, xyzc_path: str):

    # 1) 로드
    pcd_ref = load_xyz_txt(xyz_path, drop_extra=True)
    pcd_seg = load_xyz_txt(xyzc_path, drop_extra=True)

    print(f"· ref  points: {np.asarray(pcd_ref.points).shape[0]:,}")
    print(f"· segm points: {np.asarray(pcd_seg.points).shape[0]:,}")

    # 2) 스케일·시프트 초기 정합 (AABB 기반)
    aabb_ref = pcd_ref.get_axis_aligned_bounding_box()
    aabb_seg = pcd_seg.get_axis_aligned_bounding_box()

    scale_init = (
        np.linalg.norm(aabb_ref.get_extent()) / np.linalg.norm(aabb_seg.get_extent())
    )
    pcd_seg.scale(scale_init, center=(0, 0, 0))

    # 중심 맞추기
    pcd_seg.translate(pcd_ref.get_center() - pcd_seg.get_center())

    # 3) ICP (Similarity - with scaling)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100
    )
    icp = o3d.pipelines.registration.registration_icp(
        pcd_seg,
        pcd_ref,
        max_correspondence_distance=5.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=True
        ),
        criteria=criteria,
    )
    pcd_seg.transform(icp.transformation)
    print("· ICP fitness :", icp.fitness, "  RMSE :", icp.inlier_rmse)

    # 4) Poisson 메쉬 재구성
    mesh_ref = poisson_reconstruct(pcd_ref)
    mesh_seg = poisson_reconstruct(pcd_seg)

    area_ref, vol_ref = mesh_area_volume(mesh_ref)
    area_seg, vol_seg = mesh_area_volume(mesh_seg)

    # 5) 출력
    print("\n==== Surface Area & Volume (mm² / mm³) ====")
    print(f"GT  area: {area_ref:,.3f}   |   volume: {vol_ref:,.3f}")
    print(f"SEG area: {area_seg:,.3f}   |   volume: {vol_seg:,.3f}")

    dA = (area_seg - area_ref) / area_ref * 100.0
    dV = (vol_seg - vol_ref) / vol_ref * 100.0
    print("-------------------------------------------")
    print(f"ΔArea  : {dA:+.2f} %")
    print(f"ΔVolume: {dV:+.2f} %")

    # 6) (선택) 결과 mesh 저장
    # out_dir = "cmp_results"
    # os.makedirs(out_dir, exist_ok=True)
    # o3d.io.write_triangle_mesh(os.path.join(out_dir, "ref_mesh.ply"), mesh_ref)
    # o3d.io.write_triangle_mesh(os.path.join(out_dir, "seg_mesh_aligned.ply"), mesh_seg)
    # print(f"\n· 정합된 메쉬를 '{out_dir}/' 폴더에 저장했습니다.")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="XYZ vs XYZC 기하학 비교 (표면적·부피)", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--xyz", required=True, help="원본 XYZ 파일 경로")
    ap.add_argument("--xyzc", required=True, help="세그먼트 XYZC 파일 경로")
    args = ap.parse_args()

    main(args.xyz, args.xyzc)
