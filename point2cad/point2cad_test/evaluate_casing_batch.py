#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python '.\evaluate_casing_batch .py' --path1 "../assets/ply/casing/ply_validation/casing.ply"    --path2 "../assets/ply/casing/ply_test" --start 1 --end 120 --pattern "mesh_casing_{:03d}.ply"    --scale 1 --log "evaluation_casing.log"

import os
import argparse
import numpy as np
import open3d as o3d
from datetime import datetime

# ----------------------- IO & 전처리 -----------------------
def load_and_process_ply(path, scale=1.0, shift_x=0.0, rotate_deg_x=0.0):
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        raise ValueError(f"메쉬가 비어있습니다: {path}")
    mesh.compute_vertex_normals()
    if scale != 1.0:
        mesh.scale(scale, center=(0, 0, 0))
    if shift_x != 0.0:
        mesh.translate((shift_x, 0, 0))
    if rotate_deg_x != 0.0:
        radians = np.deg2rad(rotate_deg_x)
        R = mesh.get_rotation_matrix_from_axis_angle([radians, 0, 0])
        mesh.rotate(R, center=(0, 0, 0))
    return mesh

# ----------------------- 지표 계산 -----------------------
def _deg_from_rotmat(R):
    t = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(t))

def compute_absolute_position_accuracy(mesh1, mesh2, scale=1.0):
    aabb1, aabb2 = mesh1.get_axis_aligned_bounding_box(), mesh2.get_axis_aligned_bounding_box()
    c1, c2 = np.asarray(aabb1.get_center()), np.asarray(aabb2.get_center())
    trans_vec_mm = (c1 - c2) / scale
    abs_mean_mm = float(np.mean(np.abs(trans_vec_mm)))
    abs_max_mm  = float(np.max(np.abs(trans_vec_mm)))

    obb1, obb2 = mesh1.get_oriented_bounding_box(), mesh2.get_oriented_bounding_box()
    R_rel = obb2.R.T @ obb1.R
    rot_err_deg = _deg_from_rotmat(R_rel)

    return {
        "translation_vector_mm": trans_vec_mm,
        "abs_mean_mm": abs_mean_mm,
        "abs_max_mm":  abs_max_mm,
        "rotation_error_deg": rot_err_deg
    }

def compute_relative_position_accuracy_icp(mesh1, mesh2, scale=1.0,
                                           n_points=20000, icp_thresh_mm=1.0):
    pcd1 = mesh1.sample_points_uniformly(number_of_points=int(n_points))
    pcd2 = mesh2.sample_points_uniformly(number_of_points=int(n_points))

    T0 = np.eye(4)
    T0[:3, 3] = (mesh2.get_axis_aligned_bounding_box().get_center()
                 - mesh1.get_axis_aligned_bounding_box().get_center())
    pcd1.transform(T0)

    threshold = icp_thresh_mm * scale
    result = o3d.pipelines.registration.registration_icp(
        pcd1, pcd2, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    pcd1.transform(result.transformation)

    d12 = np.asarray(pcd1.compute_point_cloud_distance(pcd2)) / scale
    d21 = np.asarray(pcd2.compute_point_cloud_distance(pcd1)) / scale
    d = np.concatenate([d12, d21])
    return {"rel_mean_mm": float(np.mean(d)), "rel_max_mm": float(np.max(d))}

def compute_relative_length_accuracy(mesh1, mesh2, scale=1.0):
    b1, b2 = mesh1.get_oriented_bounding_box(), mesh2.get_oriented_bounding_box()
    size1_mm = b1.extent / scale
    size2_mm = b2.extent / scale
    diff_mm  = np.abs(size1_mm - size2_mm)
    area1_mm2 = mesh1.get_surface_area() / (scale * scale)
    area2_mm2 = mesh2.get_surface_area() / (scale * scale)
    return {"size1_mm": size1_mm, "size2_mm": size2_mm,
            "size_abs_diff_mm": diff_mm, "area1_mm2": area1_mm2, "area2_mm2": area2_mm2}

def compute_geometric_accuracy(mesh1, mesh2, n_points=10000, scale=1.0):
    pcd1 = mesh1.sample_points_uniformly(number_of_points=int(n_points))
    pcd2 = mesh2.sample_points_uniformly(number_of_points=int(n_points))
    d1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2)) / scale
    d2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1)) / scale
    chamfer = float((np.mean(d1) + np.mean(d2)) / 2)
    hausdorff = float(max(np.max(d1), np.max(d2)))
    return chamfer, hausdorff

def compute_surface_matching_accuracy(mesh1, mesh2, n_points=10000, scale=1.0):
    pcd_s = mesh1.sample_points_uniformly(number_of_points=int(n_points))
    pcd_t = mesh2.sample_points_uniformly(number_of_points=int(n_points))
    d = np.asarray(pcd_s.compute_point_cloud_distance(pcd_t)) / scale
    return float(np.mean(d)), float(np.max(d))

# ---- AABB 축길이 기반 '정확한' 부피 오차율: (rx*ry*rz - 1)*100 ----
def aabb_axis_lengths_mm(mesh, scale):
    return mesh.get_axis_aligned_bounding_box().get_extent() / scale  # [Lx,Ly,Lz] in mm

def volume_error_percent_from_axes(mesh1, mesh2, scale, eps=1e-12):
    L1 = aabb_axis_lengths_mm(mesh1, scale)
    L2 = aabb_axis_lengths_mm(mesh2, scale)

    # 0 길이 축 방지(극히 드뭄): eps로 가드
    safe_L1 = np.where(np.abs(L1) < eps, np.nan, L1)
    ratios = L2 / safe_L1  # [rx, ry, rz]
    prod_ratio = float(np.nanprod(ratios))
    err_percent_signed = (prod_ratio - 1.0) * 100.0          # 부호 보존
    err_percent_abs    = abs(err_percent_signed)             # 판정용
    approx_percent     = float(np.nansum((L2 - L1) / safe_L1)) * 100.0  # 소오차 근사

    return {
        "L1_mm": L1, "L2_mm": L2, "ratios": ratios,
        "err_percent": err_percent_signed,        # 정확식 (부호 있음)
        "err_percent_abs": err_percent_abs,       # 절대값 (통과/실패 판정)
        "approx_percent": approx_percent          # 작은 오차 근사(참고)
    }

# ----------------------- 로깅 -----------------------
def save_evaluation_log(lines, log_path):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# ----------------------- 메인 루프 -----------------------
def evaluate_pair(path1, path2, scale, shift_x, rot1, rot2,
                  icp_points, icp_thresh_mm, geom_points):
    mesh1 = load_and_process_ply(path1, scale=1.0, shift_x=shift_x, rotate_deg_x=rot1)
    mesh2 = load_and_process_ply(path2, scale=1.0, shift_x=0.0,     rotate_deg_x=rot2)

    res_abs = compute_absolute_position_accuracy(mesh1, mesh2, scale=scale)
    res_rel = compute_relative_position_accuracy_icp(
        mesh1, mesh2, scale=scale, n_points=icp_points, icp_thresh_mm=icp_thresh_mm
    )
    res_len = compute_relative_length_accuracy(mesh1, mesh2, scale=scale)
    chamfer_mm, hausdorff_mm = compute_geometric_accuracy(
        mesh1, mesh2, n_points=geom_points, scale=scale
    )
    mean_surface_mm, max_surface_mm = compute_surface_matching_accuracy(
        mesh1, mesh2, n_points=geom_points, scale=scale
    )
    
    vol_err = volume_error_percent_from_axes(mesh1, mesh2, scale=scale)

    return {
        "res_abs": res_abs, "res_rel": res_rel, "res_len": res_len,
        "chamfer_mm": chamfer_mm, "hausdorff_mm": hausdorff_mm,
        "mean_surface_mm": mean_surface_mm, "max_surface_mm": max_surface_mm,
        "vol_err": vol_err
    }

def main():
    ap = argparse.ArgumentParser(description="Batch evaluate casing meshes")
    ap.add_argument("--path1", help="기준(원본) PLY 파일 경로", default="/mnt/nas4/junewookang/point2cad/assets/casing/ply_validation/casing.ply")
    ap.add_argument("--path2", help="mesh_###.ply들이 있는 폴더", default="/mnt/nas4/junewookang/point2cad/assets/casing/ply_test")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end",   type=int, default=120)
    ap.add_argument("--pattern", default="casing_{:03d}.ply")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--shift_x", type=float, default=0.0)
    ap.add_argument("--rot_x_deg1", type=float, default=0.0)
    ap.add_argument("--rot_x_deg2", type=float, default=0.0)
    ap.add_argument("--icp_points", type=int, default=20000)
    ap.add_argument("--icp_thresh_mm", type=float, default=1.0)
    ap.add_argument("--geom_points", type=int, default=10000)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_log_path = os.path.join(script_dir, "evaluation_casing.log")
    ap.add_argument("--log", default=default_log_path)
    args = ap.parse_args()

    path1 = args.path1
    assert os.path.exists(path1), f"PATH1 not found: {path1}"

    total_count = 0
    pass_count = 0
    fail_count = 0
    file_results = []

    for i in range(args.start, args.end + 1):
        fname = args.pattern.format(i)
        path2 = os.path.join(args.path2, fname)
        if not os.path.exists(path2):
            print(f"[SKIP] not found: {path2}")
            continue

        total_count += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hdr = f"{'시간':<19} | {'측정 대상':<30} | {'측정 값':>12}"
        bar = "-" * len(hdr)

        print(f"\n>>> Evaluating: {os.path.basename(path2)}")
        try:
            R = evaluate_pair(
                path1, path2,
                scale=args.scale, shift_x=args.shift_x,
                rot1=args.rot_x_deg1, rot2=args.rot_x_deg2,
                icp_points=args.icp_points, icp_thresh_mm=args.icp_thresh_mm,
                geom_points=args.geom_points
            )

            criteria = {
                "절대 위치 (평균)": (R["res_abs"]["abs_mean_mm"], 0.10),
                "절대 위치 (최대)": (R["res_abs"]["abs_max_mm"],  0.30),
                "상대 위치 (평균)": (R["res_rel"]["rel_mean_mm"], 0.05),
                "상대 위치 (최대)": (R["res_rel"]["rel_max_mm"],  0.10),
                "길이 정확도 (X)": (abs(R["res_len"]["size1_mm"][0]-R["res_len"]["size2_mm"][0]), 0.05),
                "길이 정확도 (Y)": (abs(R["res_len"]["size1_mm"][1]-R["res_len"]["size2_mm"][1]), 0.05),
                "길이 정확도 (Z)": (abs(R["res_len"]["size1_mm"][2]-R["res_len"]["size2_mm"][2]), 0.05),
                "기하학 (평균)":   (R["chamfer_mm"], 0.10),
                "기하학 (최대)":   (R["hausdorff_mm"], 0.20),
                "표면 일치 (평균)": (R["mean_surface_mm"], 0.10),
                "표면 일치 (최대)": (R["max_surface_mm"],  0.30),
                "부피 오차(%)":    (R["vol_err"]["err_percent_abs"], 1.0),
            }

            rows = [hdr, bar]
            fail_notes = []
            display_map = {
                "부피 오차(%)": R["vol_err"]["err_percent"]
            }

            is_fail = False
            for metric, (val, thr) in criteria.items():
                show_val = display_map.get(metric, val)
                rows.append(f"{now:<19} | {metric:<30} | {show_val:>12.6f}")
                if val > thr:
                    ref_thr = thr
                    fail_notes.append(f"- {metric}: {show_val:.6f} (기준 {ref_thr})")
                    is_fail = True

            if fail_notes:
                rows.append("")
                rows.append("특이사항:")
                rows.extend(fail_notes)
                fail_count += 1
                file_results.append(f"FAIL: {os.path.basename(path2)}")
            else:
                rows.append("")
                rows.append("특이사항: 없음")
                pass_count += 1
                file_results.append(f"PASS: {os.path.basename(path2)}")

            head = [
                "====================",
                f"Mesh1(ref): {os.path.basename(path1)}",
                f"Mesh2(test): {os.path.basename(path2)}",
                f"평가 시각: {now}",
            ]
            out_lines = head + rows

            print("\n".join(rows))
            save_evaluation_log(out_lines, args.log)

        except Exception as e:
            print(f"[ERROR] {path2}: {e}")
            fail_count += 1
            file_results.append(f"ERROR: {os.path.basename(path2)}: {e}")

    # Write summary to log
    summary_lines = [
        "\n==================== SUMMARY ====================",
        f"총 평가 파일: {total_count}",
        f"통과(PASS): {pass_count}",
        f"실패(FAIL): {fail_count}",
        "-----------------------------------------------",
    ]
    summary_lines.extend(file_results)
    save_evaluation_log(summary_lines, args.log)

if __name__ == "__main__":
    main()
