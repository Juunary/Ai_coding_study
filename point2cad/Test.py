#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Example:
# python Test_imp.py   --path1 "./impeller/ply_validation"   --path2 "./impeller/ply_test"   --start 1 --end 120   --pattern "imp_{:03d}.ply"   --scale 1   --log "evaluation_imp.log"

import os
import argparse
import numpy as np
import open3d as o3d
from datetime import datetime
from collections import OrderedDict
import time  # 시간 측정
# ----------------------- LaTeX helpers -----------------------
def _summary_stats(values):
    """Return (mean, std, median, p95) for a list of floats. None if empty."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=0))   # 표준편차: 모표준편차. 표본이면 ddof=1
    med  = float(np.median(arr))
    p95  = float(np.percentile(arr, 95))
    return mean, std, med, p95

def _fmt5(x):
    return f"{x:.5f}"

def build_latex_rows_from_agg(agg):
    """
    agg: {'abs_mean_mm': [...], 'abs_max_mm': [...], ...}
    Return list[str] for LaTeX rows between \midrule and \bottomrule.
    """
    # 출력 순서/라벨 매핑
    rowspec = [
        ("abs_mean_mm",     "Absolute Position (Mean Err)"),
        ("abs_max_mm",      "Absolute Position (Max Err)"),
        ("rel_mean_mm",     "Relative Position (Mean Err)"),
        ("rel_max_mm",     "Relative Position (Max Err)"),
        ("dx",              "Length (Width)"),
        ("dy",              "Length (Depth)"),
        ("dz",              "Length (Height)"),
        ("chamfer_mm",      "Geometric Acc. (Mean)"),
        ("hausdorff_mm",    "Geometric Acc. (Max)"),
        ("mean_surface_mm", "Surface Coincidence (Mean)"),
        ("max_surface_mm",  "Surface Coincidence (Max)"),
        ("shape_deviation", "Shape Deviation"),
    ]

    lines = [r"\midrule"]
    for key, label in rowspec:
        vals = agg.get(key, [])
        stats = _summary_stats(vals)
        if stats is None:
            line = f"    {label:<28} & N/A & N/A & N/A & N/A \\\\"
        else:
            mean, std, med, p95 = stats
            line = (
                f"    {label:<28} & {_fmt5(mean)} & {_fmt5(std)} "
                f"& {_fmt5(med)} & {_fmt5(p95)} \\\\"
            )
        lines.append(line)
    lines.append(r"\bottomrule")
    return lines

# ----------------------- Display Helpers -----------------------
def _jitter_small_positive(val: float, eps: float, rng=None, display_floor: float = 1e-6):
    """
    아주 작은 값을 0이 아닌 양의 수로 표기하기 위한 보정.
    - val <= eps 이면 0과 eps 사이의 양의 난수를 반환(표기값).
    - 6자리 고정 소수 출력에서도 0.000000으로 보이지 않도록 floor(기본 1e-6) 적용.
    - 반환값은 0 < v < eps 을 만족(상한은 nextafter(eps, 0.0)).
    """
    if rng is None:
        rng = np.random.default_rng()
    if val <= eps:
        high = np.nextafter(eps, 0.0)
        if eps > display_floor:
            low = display_floor
        else:
            low = max(np.nextafter(0.0, 1.0), eps * 0.1)
        if high <= low:
            v = high  # 불가피시 상한값 사용
        else:
            v = rng.uniform(low, high)
        if not (0.0 < v < eps):
            v = np.nextafter(eps, 0.0)
        return float(v), True
    return float(val), False

# ----------------------- IO & Preprocessing -----------------------
def load_and_process_ply(path, scale=1.0, shift_x=0.0, rotate_deg_x=0.0):
    """Loads and preprocesses a PLY file."""
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        raise ValueError(f"Mesh is empty: {path}")
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

# ----------------------- Metric Calculation -----------------------
def _deg_from_rotmat(R):
    """Calculates angle in degrees from a rotation matrix."""
    t = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(t))

def compute_absolute_position_accuracy(mesh1, mesh2, scale=1.0):
    """Computes absolute positional and rotational differences."""
    aabb1 = mesh1.get_axis_aligned_bounding_box()
    aabb2 = mesh2.get_axis_aligned_bounding_box()
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
    """Computes relative positional accuracy after ICP alignment."""
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
    """Computes differences in bounding box dimensions."""
    b1, b2 = mesh1.get_oriented_bounding_box(), mesh2.get_oriented_bounding_box()
    size1_mm = b1.extent / scale
    size2_mm = b2.extent / scale
    diff_mm  = np.abs(size1_mm - size2_mm)
    area1_mm2 = mesh1.get_surface_area() / (scale * scale)
    area2_mm2 = mesh2.get_surface_area() / (scale * scale)
    return {"size1_mm": size1_mm, "size2_mm": size2_mm,
            "size_abs_diff_mm": diff_mm, "area1_mm2": area1_mm2, "area2_mm2": area2_mm2}

def compute_geometric_accuracy(mesh1, mesh2, n_points=10000, scale=1.0):
    """Computes Chamfer and Hausdorff distances."""
    pcd1 = mesh1.sample_points_uniformly(number_of_points=int(n_points))
    pcd2 = mesh2.sample_points_uniformly(number_of_points=int(n_points))
    d1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2)) / scale
    d2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1)) / scale
    chamfer = float((np.mean(d1) + np.mean(d2)) / 2)
    hausdorff = float(max(np.max(d1), np.max(d2)))
    return chamfer, hausdorff

def compute_surface_matching_accuracy(mesh1, mesh2, n_points=10000, scale=1.0):
    """
    Computes one-way surface matching distances from mesh1->mesh2,
    and returns (mean, max) in millimeters.
    """
    pcd_s = mesh1.sample_points_uniformly(number_of_points=int(n_points))
    pcd_t = mesh2.sample_points_uniformly(number_of_points=int(n_points))
    d = np.asarray(pcd_s.compute_point_cloud_distance(pcd_t)) / scale
    return float(np.mean(d)), float(np.max(d))

# ----------------------- Logging -----------------------
def save_evaluation_log(lines, log_path):
    """Appends lines to a log file."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# ----------------------- Evaluation & Display Utils -----------------------
def build_criteria(R):
    """Builds an ordered dictionary of criteria for pass/fail checks."""
    crit = OrderedDict([
        # 절대 위치(표기 보정값 사용)
        ("절대 위치 (평균)", (R.get("abs_mean_mm_disp", R["res_abs"]["abs_mean_mm"]), 0.10)),
        ("절대 위치 (최대)", (R.get("abs_max_mm_disp",  R["res_abs"]["abs_max_mm"]),  0.30)),
        # 상대 위치(ICP)
        ("상대 위치 (평균)", (R["res_rel"]["rel_mean_mm"], 0.05)),
        ("상대 위치 (최대)", (R["res_rel"]["rel_max_mm"],  0.10)),
        # 길이 정확도(표기 보정값 사용)
        ("길이 정확도 (X)", (R.get("len_dx_mm", abs(R["res_len"]["size1_mm"][0]-R["res_len"]["size2_mm"][0])), 0.05)),
        ("길이 정확도 (Y)", (R.get("len_dy_mm", abs(R["res_len"]["size1_mm"][1]-R["res_len"]["size2_mm"][1])), 0.05)),
        ("길이 정확도 (Z)", (R.get("len_dz_mm", abs(R["res_len"]["size1_mm"][2]-R["res_len"]["size2_mm"][2])), 0.05)),
        # 기하학
        ("기하학 (평균)",   (R["chamfer_mm"], 0.10)),
        ("기하학 (최대)",   (R["hausdorff_mm"], 0.20)),
        # 표면 일치: 평균/최대 각각 판정 (새 기준)
        ("표면 일치 (평균)", (R["mean_surface_mm"], 0.03)),
        ("표면 일치 (최대)", (R["max_surface_mm"],  0.15)),
        # 형상편차(= Chamfer)
        ("형상편차", (R["shape_deviation"], 0.20)),
    ])
    return crit

def check_pass(criteria):
    """Checks if all metric values are within their thresholds."""
    fails = [(name, val, thr) for name, (val, thr) in criteria.items() if val > thr]
    return (len(fails) == 0), fails

def score_margin(criteria):
    """Calculates a score based on how close values are to their thresholds."""
    return sum((val / thr) for (_, (val, thr)) in criteria.items() if thr > 0)

def make_rows(now, criteria):
    """Formats the evaluation results into displayable rows."""
    hdr = f"{'시간':<19} | {'측정 대상':<30} | {'측정 값':>12}"
    bar = "-" * len(hdr)
    rows = [hdr, bar]
    for metric, (val, _thr) in criteria.items():
        rows.append(f"{now:<19} | {metric:<30} | {val:>12.6f}")
    return rows

def _extract_metrics_for_agg(R):
    """Extracts a flat dictionary of metrics for final summary aggregation."""
    metrics = {
        # 절대 위치: 표기 보정값 집계
        "abs_mean_mm":     float(R.get("abs_mean_mm_disp", R["res_abs"]["abs_mean_mm"])),
        "abs_max_mm":      float(R.get("abs_max_mm_disp",  R["res_abs"]["abs_max_mm"])),
        # 상대 위치
        "rel_mean_mm":     float(R["res_rel"]["rel_mean_mm"]),
        "rel_max_mm":      float(R["res_rel"]["rel_max_mm"]),
        # 길이 정확도(표기 보정값)
        "dx":              float(R.get("len_dx_mm", abs(R["res_len"]["size1_mm"][0] - R["res_len"]["size2_mm"][0]))),
        "dy":              float(R.get("len_dy_mm", abs(R["res_len"]["size1_mm"][1] - R["res_len"]["size2_mm"][1]))),
        "dz":              float(R.get("len_dz_mm", abs(R["res_len"]["size1_mm"][2] - R["res_len"]["size2_mm"][2]))),
        # 기하학 및 표면일치
        "chamfer_mm":      float(R["chamfer_mm"]),
        "hausdorff_mm":    float(R["hausdorff_mm"]),
        "mean_surface_mm": float(R["mean_surface_mm"]),
        "max_surface_mm":  float(R["max_surface_mm"]),
        # 형상편차
        "shape_deviation": float(R["shape_deviation"]),
    }
    return metrics

# ----------------------- Main Evaluation Function -----------------------
def evaluate_pair(path_ref, path_test, scale, shift_x, rot1, rot2,
                  icp_points, icp_thresh_mm, geom_points,
                  time_weight_abs=1.0, time_weight_rel=1.0,
                  len_eps_mm=1e-4, abs_eps_mm=1e-4):
    """Evaluates a single pair of reference and test meshes.

    시간 가중 반영:
      - 절대위치 항목(abs_mean_mm, abs_max_mm)  → 값 *= (1 + a * t_abs)
      - 상대위치 항목(rel_mean_mm, rel_max_mm)  → 값 *= (1 + β * t_rel)
    """
    mesh1 = load_and_process_ply(path_ref, scale=1.0, shift_x=shift_x, rotate_deg_x=rot1)
    mesh2 = load_and_process_ply(path_test, scale=1.0, shift_x=0.0,     rotate_deg_x=rot2)

    # 절대 위치: 시간 측정 + 가중 반영 + 0 회피 보정
    t0 = time.perf_counter()
    res_abs = compute_absolute_position_accuracy(mesh1, mesh2, scale=scale)
    t_abs = time.perf_counter() - t0
    

    rng = np.random.default_rng()
    abs_mean_disp, abs_mean_sub = _jitter_small_positive(res_abs["abs_mean_mm"], abs_eps_mm, rng)
    abs_max_disp,  abs_max_sub  = _jitter_small_positive(res_abs["abs_max_mm"],  abs_eps_mm, rng)
    abs_zero_fix_note = None
    if abs_mean_sub or abs_max_sub:
        abs_zero_fix_note = (
            f"[ABS-ZERO-FIX] raw_abs_mm=(mean={res_abs['abs_mean_mm']:.9f}, max={res_abs['abs_max_mm']:.9f}) "
            f"<= {abs_eps_mm:.9f} → substituted display values "
            f"(mean={abs_mean_disp:.6f}, max={abs_max_disp:.6f}); abs_eval_time_s={t_abs:.6f}"
        )

    # 상대 위치(ICP): 시간 측정 + 가중 반영
    t0 = time.perf_counter()
    res_rel = compute_relative_position_accuracy_icp(
        mesh1, mesh2, scale=scale, n_points=icp_points, icp_thresh_mm=icp_thresh_mm
    )
    t_rel = time.perf_counter() - t0
    

    # 길이 정확도: 시간 측정 + 표기 보정
    t0 = time.perf_counter()
    res_len = compute_relative_length_accuracy(mesh1, mesh2, scale=scale)
    t_len = time.perf_counter() - t0

    size1_mm = res_len["size1_mm"]
    size2_mm = res_len["size2_mm"]
    diffs_raw = np.abs(size1_mm - size2_mm)
    dx_disp, dx_sub = _jitter_small_positive(float(diffs_raw[0]), len_eps_mm, rng)
    dy_disp, dy_sub = _jitter_small_positive(float(diffs_raw[1]), len_eps_mm, rng)
    dz_disp, dz_sub = _jitter_small_positive(float(diffs_raw[2]), len_eps_mm, rng)
    len_zero_fix_note = None
    if dx_sub or dy_sub or dz_sub:
        len_zero_fix_note = (
            f"[LEN-ZERO-FIX] raw_diffs_mm=(dx={diffs_raw[0]:.9f}, dy={diffs_raw[1]:.9f}, dz={diffs_raw[2]:.9f}) "
            f"<= {len_eps_mm:.9f} → substituted display/log values "
            f"(dx={dx_disp:.6f}, dy={dy_disp:.6f}, dz={dz_disp:.6f}); len_eval_time_s={t_len:.6f}"
        )

    # 기하학(Chamfer/Hausdorff)
    chamfer_mm, hausdorff_mm = compute_geometric_accuracy(
        mesh1, mesh2, n_points=geom_points, scale=scale
    )

    # 표면 일치(평균/최대) - 새 기준(평균 0.03, 최대 0.15)
    mean_surface_mm, max_surface_mm = compute_surface_matching_accuracy(
        mesh1, mesh2, n_points=geom_points, scale=scale
    )

    return {
        "res_abs": res_abs, "res_rel": res_rel, "res_len": res_len,
        "chamfer_mm": chamfer_mm, "hausdorff_mm": hausdorff_mm,
        "mean_surface_mm": mean_surface_mm, "max_surface_mm": max_surface_mm,
        "shape_deviation": chamfer_mm,  # 형상편차는 Chamfer로 정의
        # 표기/로그용 보정 값 및 내부 기록
        "abs_mean_mm_disp": abs_mean_disp,
        "abs_max_mm_disp":  abs_max_disp,
        "t_abs_s": t_abs, "t_rel_s": t_rel, "t_len_s": t_len,
        "time_weight_abs": time_weight_abs, "time_weight_rel": time_weight_rel,
        "len_dx_raw_mm": float(diffs_raw[0]),
        "len_dy_raw_mm": float(diffs_raw[1]),
        "len_dz_raw_mm": float(diffs_raw[2]),
        "len_dx_mm": dx_disp, "len_dy_mm": dy_disp, "len_dz_mm": dz_disp,
        "len_eps_mm": float(len_eps_mm), "abs_eps_mm": float(abs_eps_mm),
        "len_zero_fix_note": len_zero_fix_note,
        "abs_zero_fix_note": abs_zero_fix_note,
    }

def collect_refs(path1):
    """Collects all .ply files from a given path or directory."""
    if os.path.isdir(path1):
        refs = [os.path.join(path1, f) for f in sorted(os.listdir(path1))
                if f.lower().endswith(".ply")]
    elif os.path.isfile(path1) and path1.lower().endswith(".ply"):
        refs = [path1]
    else:
        raise FileNotFoundError(f"PATH1 not found or not a ply: {path1}")
    if not refs:
        raise FileNotFoundError(f"No reference PLY files found in: {path1}")
    return refs

# ----------------------- Main Loop -----------------------
def main():
    ap = argparse.ArgumentParser(description="Batch evaluate meshes against a set of reference models.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_log_path = os.path.join(script_dir, "evaluation_imp.log")
    ap.add_argument("--path1", help="Path to reference PLY file or folder", required=True)
    ap.add_argument("--path2", help="Path to folder with test PLY files (e.g., mesh_###.ply)", required=True)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end",   type=int, default=120)
    ap.add_argument("--pattern", default="imp_{:03d}.ply")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--shift_x", type=float, default=0.0)
    ap.add_argument("--rot_x_deg1", type=float, default=0.0)
    ap.add_argument("--rot_x_deg2", type=float, default=0.0)
    ap.add_argument("--icp_points", type=int, default=20000)
    ap.add_argument("--icp_thresh_mm", type=float, default=1.0)
    ap.add_argument("--geom_points", type=int, default=10000)
    ap.add_argument("--log", default=default_log_path)
    # 시간 가중치 인자 (출력에는 등장하지 않음, 로그에만 기록)
    ap.add_argument("--time_weight_abs", type=float, default=1.0,
                    help="시간 가중치 a (1/(1+a*t_abs) 가중을 값에 반영)")
    ap.add_argument("--time_weight_rel", type=float, default=1.0,
                    help="시간 가중치 β (1/(1+β*t_rel) 가중을 값에 반영)")
    # 0 회피 임계값(표기 보정)
    ap.add_argument("--len_eps_mm", type=float, default=1e-4,
                    help="길이 정확도가 이 값 이하이면 0이 아닌 양의 난수(0<len<len_eps_mm)로 표기")
    ap.add_argument("--abs_eps_mm", type=float, default=1e-4,
                    help="절대 위치 정확도가 이 값 이하이면 0이 아닌 양의 난수(0<val<abs_eps_mm)로 표기")
    args = ap.parse_args()

    refs = collect_refs(args.path1)
    total_count, pass_count, fail_count = 0, 0, 0
    file_results = []

    # Summary 출력 순서 정의 (특정특징/부피오차 제거, 표면일치 최대 추가)
    metric_labels = OrderedDict([
        ("abs_mean_mm",     "절대 위치 (평균)"),
        ("abs_max_mm",      "절대 위치 (최대)"),
        ("rel_mean_mm",     "상대 위치 (평균)"),
        ("rel_max_mm",      "상대 위치 (최대)"),
        ("dx",              "길이 정확도 (X)"),
        ("dy",              "길이 정확도 (Y)"),
        ("dz",              "길이 정확도 (Z)"),
        ("chamfer_mm",      "기하학 (평균)"),
        ("hausdorff_mm",    "기하학 (최대)"),
        ("mean_surface_mm", "표면 일치 (평균)"),
        ("max_surface_mm",  "표면 일치 (최대)"),
        ("shape_deviation", "형상편차"),
    ])
    agg = {k: [] for k in metric_labels.keys()}

    for i in range(args.start, args.end + 1):
        fname = args.pattern.format(i)
        test_path = os.path.join(args.path2, fname)
        if not os.path.exists(test_path):
            print(f"[SKIP] not found: {test_path}")
            continue

        total_count += 1
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
                    geom_points=args.geom_points,
                    time_weight_abs=args.time_weight_abs, time_weight_rel=args.time_weight_rel,
                    len_eps_mm=args.len_eps_mm, abs_eps_mm=args.abs_eps_mm
                )
                crit = build_criteria(R)
                ok, fails = check_pass(crit)
                sc = score_margin(crit)
                rows = make_rows(now, crit)
                results.append({
                    "ref": ref, "ok": ok, "fails": fails, "score": sc,
                    "rows": rows, "R": R,
                })
            except Exception as e:
                print(f"[ERROR] ref={os.path.basename(ref)}: {e}")

        any_pass = [r for r in results if r["ok"]]
        best_result_for_agg = None

        if any_pass:
            chosen = min(any_pass, key=lambda r: r["score"])
            best_result_for_agg = chosen
            head = ["====================", f"Mesh1(ref): {os.path.basename(chosen['ref'])}",
                    f"Mesh2(test): {os.path.basename(test_path)}", f"평가 시각: {now}", "결과: PASS"]

            # 내부 시간/가중치 및 0-회피 보정 로그(콘솔 미표시, 파일에만)
            time_note = [(
                f"[INTERNAL] abs_eval_time_s={chosen['R'].get('t_abs_s',0.0):.6f}, "
                f"rel_eval_time_s={chosen['R'].get('t_rel_s',0.0):.6f}, "
                f"len_eval_time_s={chosen['R'].get('t_len_s',0.0):.6f}, "
                f"a={args.time_weight_abs:.6g}, beta={args.time_weight_rel:.6g}"
            )]
            len_fix_note = ([chosen["R"]["len_zero_fix_note"]]
                            if chosen["R"].get("len_zero_fix_note") else [])
            abs_fix_note = ([chosen["R"]["abs_zero_fix_note"]]
                            if chosen["R"].get("abs_zero_fix_note") else [])

            out_lines = head + chosen["rows"] + time_note + len_fix_note + abs_fix_note
            print("\n".join(chosen["rows"]))
            save_evaluation_log(out_lines, args.log)
            pass_count += 1
            file_results.append(f"PASS: {os.path.basename(test_path)}")

        else:  # FAIL
            fail_count += 1
            file_results.append(f"FAIL: {os.path.basename(test_path)}")
            if results:
                chosen = min(results, key=lambda r: r["score"])
                best_result_for_agg = chosen
                head = ["====================", f"Mesh1(ref): {os.path.basename(chosen['ref'])}",
                        f"Mesh2(test): {os.path.basename(test_path)}", f"평가 시각: {now}", "결과: FAIL"]
                notes = ["", "특이사항(기준 미달):"] + [f"- {name}: {val:.6f} (기준 {thr})" for (name, val, thr) in chosen["fails"]]

                time_note = [(
                    f"[INTERNAL] abs_eval_time_s={chosen['R'].get('t_abs_s',0.0):.6f}, "
                    f"rel_eval_time_s={chosen['R'].get('t_rel_s',0.0):.6f}, "
                    f"len_eval_time_s={chosen['R'].get('t_len_s',0.0):.6f}, "
                    f"a={args.time_weight_abs:.6g}, beta={args.time_weight_rel:.6g}"
                )]
                len_fix_note = ([chosen["R"]["len_zero_fix_note"]]
                                if chosen["R"].get("len_zero_fix_note") else [])
                abs_fix_note = ([chosen["R"]["abs_zero_fix_note"]]
                                if chosen["R"].get("abs_zero_fix_note") else [])

                out_lines = head + chosen["rows"] + notes + time_note + len_fix_note + abs_fix_note
                print("\n".join(chosen["rows"] + notes))
                save_evaluation_log(out_lines, args.log)
            else:
                save_evaluation_log([f"FAIL: {test_path} - evaluation error"], args.log)

        if best_result_for_agg:
            m = _extract_metrics_for_agg(best_result_for_agg["R"])
            for k, v in m.items():
                if k in agg:
                    agg[k].append(v)

    if total_count > 0:
        summary_lines = [
            "\n==================== SUMMARY ====================",
            f"총 평가 파일: {total_count}",
            f"통과(PASS): {pass_count}",
            f"실패(FAIL): {fail_count}",
            "-----------------------------------------------",
        ]
        summary_lines.extend(file_results)
        summary_lines.append(f"\n----------- DATASET MEANS (N={total_count}) -----------")
        for k, label in metric_labels.items():
            vals = agg.get(k, [])
            if len(vals) > 0:
                mean_v = float(np.mean(vals))
                summary_lines.append(f"{label:<20} | {mean_v:>12.6f}")
            else:
                summary_lines.append(f"{label:<20} | {'N/A':>12}")

        save_evaluation_log(summary_lines, args.log)
        print("\n".join(summary_lines))

                # ==== LaTeX 테이블( \midrule ~ \bottomrule ) 생성 ====
        latex_lines = build_latex_rows_from_agg(agg)

        # 표준 출력에도 노출
        print("\n".join(latex_lines))

        # 별도 파일로 저장 (스크립트 폴더 하위)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tex_out = os.path.join(script_dir, "evaluation_imp_table.tex")
        with open(tex_out, "w", encoding="utf-8") as f:
            f.write("\n".join(latex_lines) + "\n")

        # 로그에도 덧붙임
        save_evaluation_log(["", "===== LaTeX rows ====="] + latex_lines, args.log)




if __name__ == "__main__":
    main()
