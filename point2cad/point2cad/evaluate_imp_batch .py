#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 예)
# python evaluate_imp_batch.py    --path1 "..\assets\ply\impeller\4wings\ply_validation"    --path2 "..\assets\ply\impeller\4wings\ply_test"    --start 1 --end 118 --pattern "mesh_xyzc_data_{:03d}.ply"  --scale 1 --log "evaluation_imp.log"

import os
import argparse
import numpy as np
import open3d as o3d
from datetime import datetime
from collections import OrderedDict

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

def volume_from_aabb_world_mm(mesh, scale):
    L = mesh.get_axis_aligned_bounding_box().get_extent() / scale
    return float(np.prod(L)), L

def volume_error_percent_aabb(mesh1, mesh2, scale, mode="ref1"):
    V1, _ = volume_from_aabb_world_mm(mesh1, scale)
    V2, _ = volume_from_aabb_world_mm(mesh2, scale)
    dV = abs(V1 - V2)
    if mode == "ref1":
        err = dV / V1 * 100.0
    elif mode == "ref2":
        err = dV / V2 * 100.0
    else:
        err = 2.0 * dV / (V1 + V2) * 100.0
    return {"V1_mm3": V1, "V2_mm3": V2, "abs_diff_mm3": dV, "err_percent": err}

# ----------------------- 로깅 -----------------------
def save_evaluation_log(lines, log_path):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# ----------------------- 평가/표출 유틸 -----------------------
def build_criteria(R):
    # 순서 고정
    return OrderedDict([
        ("절대 위치 (평균)", (R["res_abs"]["abs_mean_mm"], 0.10)),
        ("절대 위치 (최대)", (R["res_abs"]["abs_max_mm"],  0.30)),
        ("상대 위치 (평균)", (R["res_rel"]["rel_mean_mm"], 0.05)),
        ("상대 위치 (최대)", (R["res_rel"]["rel_max_mm"],  0.10)),
        ("길이 정확도 (X)", (abs(R["res_len"]["size1_mm"][0]-R["res_len"]["size2_mm"][0]), 0.05)),
        ("길이 정확도 (Y)", (abs(R["res_len"]["size1_mm"][1]-R["res_len"]["size2_mm"][1]), 0.05)),
        ("길이 정확도 (Z)", (abs(R["res_len"]["size1_mm"][2]-R["res_len"]["size2_mm"][2]), 0.05)),
        ("기하학 (평균)",   (R["chamfer_mm"], 0.10)),
        ("기하학 (최대)",   (R["hausdorff_mm"], 0.20)),
        ("표면 일치 (평균)", (R["mean_surface_mm"], 0.10)),
        ("표면 일치 (최대)", (R["max_surface_mm"],  0.30)),
        ("부피 오차(%)",    (R["res_vol"]["err_percent"], 1.0)),
    ])

def check_pass(criteria):
    fails = [(name, val, thr) for name, (val, thr) in criteria.items() if val > thr]
    return (len(fails) == 0), fails

def score_margin(criteria):
    # 작은 값이 더 좋음 (임계 대비 비율의 합)
    return sum((val / thr) for (_, (val, thr)) in criteria.items())

def make_rows(now, criteria):
    hdr = f"{'시간':<19} | {'측정 대상':<30} | {'측정 값':>12}"
    bar = "-" * len(hdr)
    rows = [hdr, bar]
    for metric, (val, _thr) in criteria.items():
        rows.append(f"{now:<19} | {metric:<30} | {val:>12.6f}")
    return rows

# ----------------------- 평가 함수 -----------------------
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
    res_vol = volume_error_percent_aabb(mesh1, mesh2, scale=scale, mode="ref1")

    return {
        "res_abs": res_abs, "res_rel": res_rel, "res_len": res_len,
        "chamfer_mm": chamfer_mm, "hausdorff_mm": hausdorff_mm,
        "mean_surface_mm": mean_surface_mm, "max_surface_mm": max_surface_mm,
        "res_vol": res_vol
    }

# ----------------------- 메인 루프 -----------------------
def main():
    ap = argparse.ArgumentParser(description="Batch evaluate meshes using ALL validations in a folder (pass if ANY ref passes)")
    ap.add_argument("--path1", required=True, help="검증(원본) PLY 폴더(또는 단일 파일) 경로")
    ap.add_argument("--path2", required=True, help="테스트 mesh_###.ply들이 있는 폴더")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end",   type:int, default=120)
    ap.add_argument("--pattern", default="mesh_xyzc_data_{:03d}.ply")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--shift_x", type=float, default=0.0)
    ap.add_argument("--rot_x_deg1", type=float, default=0.0)  # 검증메쉬 회전(공통)
    ap.add_argument("--rot_x_deg2", type=float, default=0.0)  # 테스트메쉬 회전
    ap.add_argument("--icp_points", type=int, default=20000)
    ap.add_argument("--icp_thresh_mm", type=float, default=1.0)
    ap.add_argument("--geom_points", type=int, default=10000)
    ap.add_argument("--log", default="evaluation_imp.log")
    args = ap.parse_args()

    # 검증(ref) 목록 만들기
    if os.path.isdir(args.path1):
        refs = sorted(
            [os.path.join(args.path1, f) for f in os.listdir(args.path1)
             if f.lower().endswith(".ply")]
        )
        if not refs:
            raise FileNotFoundError(f"검증 폴더에 .ply가 없습니다: {args.path1}")
    else:
        if not os.path.exists(args.path1):
            raise FileNotFoundError(f"검증 경로가 없습니다: {args.path1}")
        refs = [args.path1]

    for i in range(args.start, args.end + 1):
        fname = args.pattern.format(i)
        test_path = os.path.join(args.path2, fname)
        if not os.path.exists(test_path):
            print(f"[SKIP] not found: {test_path}")
            continue

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n>>> Evaluating: {os.path.basename(test_path)}  (refs: {len(refs)})")

        results = []
        for ref in refs:
            try:
                R = evaluate_pair(
                    ref, test_path,
                    scale=args.scale, shift_x=args.shift_x,
                    rot1=args.rot_x_deg1, rot2=args.rot_x_deg2,
                    icp_points=args.icp_points, icp_thresh_mm=args.icp_thresh_mm,
                    geom_points=args.geom_points
                )
                crit = build_criteria(R)
                ok, fails = check_pass(crit)
                sc = score_margin(crit)
                rows = make_rows(now, crit)
                results.append({
                    "ref": ref, "crit": crit, "ok": ok, "fails": fails,
                    "score": sc, "rows": rows
                })
            except Exception as e:
                print(f"[ERROR] ref={os.path.basename(ref)}: {e}")

        # 통과 선택 로직: 하나라도 ok==True면 PASS, 그중 score 가장 낮은 ref 선택
        any_pass = [r for r in results if r["ok"]]
        if any_pass:
            chosen = min(any_pass, key=lambda r: r["score"])
            head = [
                "====================",
                f"Mesh1(ref): {os.path.basename(chosen['ref'])}",
                f"Mesh2(test): {os.path.basename(test_path)}",
                f"평가 시각: {now}",
                "결과: PASS (다수 검증 중 최소 하나 통과)",
            ]
            tail = ["", "특이사항: 없음(모두 통과)"]
            out_lines = head + chosen["rows"] + tail

            # 콘솔 & 로그
            print("\n".join(chosen["rows"]))
            print("\n".join(tail))
            save_evaluation_log(out_lines, args.log)

        else:
            # 전부 실패: 각 기준별 실패 요약을 모두 기록
            all_lines = []
            for r in results:
                head = [
                    "====================",
                    f"Mesh1(ref): {os.path.basename(r['ref'])}",
                    f"Mesh2(test): {os.path.basename(test_path)}",
                    f"평가 시각: {now}",
                    "결과: FAIL (모든 검증 기준 미달)",
                ]
                notes = ["", "특이사항(기준 미달):"] + [
                    f"- {name}: {val:.6f} (기준 {thr})" for (name, val, thr) in r["fails"]
                ]
                block = head + r["rows"] + notes
                all_lines.extend(block)
                # 콘솔에도 표시
                print("\n".join(r["rows"]))
                print("\n".join(notes))

            save_evaluation_log(all_lines, args.log)

if __name__ == "__main__":
    main()
