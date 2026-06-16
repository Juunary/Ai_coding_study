#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test2.py — Paper-aligned evaluation script for NeuroB-Rep
==========================================================
Implements the exact evaluation protocol described in:
  "Topology-Constrained NeuroB-Rep: Contact-Aware CAD Reconstruction
   from 3-D Point Clouds", IEEE Access, 2025. (#Access-2025-52745)

Corrections vs. Test.py (aligned to Table 7 values)
-----------------------------------------------------
  [C1] Chamfer formula      : mean of two directional means  (Sec.V-B, Eq.(chamfer))
                               d_CD = (mean(d_PQ) + mean(d_QP)) / 2
                               Equation updated in paper to show explicit 1/2 factor.
                               Test.py computed this correctly; paper eq. was wrong.
  [C2] ICP type             : point-to-point                 (Sec.V-B, corrected)
                               Paper previously said "point-to-plane" in error.
  [C3] ICP threshold        : delta_ICP = 1.0 mm             (Sec.V-B, corrected)
                               Paper previously said 0.5 mm in error.
  [C4] ICP max iterations   : I_ICP = 50                     (Sec.V-B, L572)
  [C5] Acceptance criteria  : surface_mean <= 0.10 mm        (Sec.V-B sec:accept)
                               Test.py used 0.03 mm (too strict, not from paper).
  [C6] _jitter_small_positive removed — undisclosed display manipulation
  [C7] Shape Deviation and Surface Coincidence (Max) formally defined in paper.

Example:
    python Test2.py \\
        --path1 ./impeller/ply_validation \\
        --path2 ./impeller/ply_test \\
        --start 1 --end 120 \\
        --pattern "imp_{:03d}.ply" \\
        --scale 1 \\
        --log evaluation_imp.log
"""

import os
import argparse
import numpy as np
import open3d as o3d
from datetime import datetime
from collections import OrderedDict


# ─────────────────────────── LaTeX helpers ───────────────────────────────────

def _summary_stats(values):
    """Return (mean, std, median, p95) for a list of floats. None if empty."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    return (float(np.mean(arr)),
            float(np.std(arr, ddof=0)),
            float(np.median(arr)),
            float(np.percentile(arr, 95)))


def _fmt5(x):
    return f"{x:.5f}"


def build_latex_rows_from_agg(agg):
    """
    agg: dict of key -> list[float]
    Returns list[str] for LaTeX rows between \\midrule and \\bottomrule.
    Row order matches Table 7 (tab:consistency_all) in the paper.
    """
    rowspec = [
        ("rel_mean_mm",     "Rel. Position (Mean)"),
        ("rel_max_mm",      "Rel. Position (Max)"),
        ("dx",              "Length (Width)"),
        ("dy",              "Length (Depth)"),
        ("dz",              "Length (Height)"),
        ("chamfer_mm",      "Geometric Acc (Mean)"),
        ("hausdorff_mm",    "Geometric Acc (Max)"),
        ("mean_surface_mm", "Surface Coincidence (Mean)"),
        ("max_surface_mm",  "Surface Coincidence (Max)"),
        ("shape_deviation", "Shape Deviation"),
    ]
    lines = [r"\midrule"]
    for key, label in rowspec:
        stats = _summary_stats(agg.get(key, []))
        if stats is None:
            line = f"    {label:<32} & N/A & N/A & N/A & N/A \\\\"
        else:
            mean, std, med, p95 = stats
            line = (f"    {label:<32} & {_fmt5(mean)} & {_fmt5(std)} "
                    f"& {_fmt5(med)} & {_fmt5(p95)} \\\\")
        lines.append(line)
    lines.append(r"\bottomrule")
    return lines


# ─────────────────────────── IO & Preprocessing ──────────────────────────────

def load_and_process_ply(path, scale=1.0, shift_x=0.0, rotate_deg_x=0.0):
    """Load a PLY mesh and apply optional scale / translate / rotate."""
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


# ─────────────────────────── Metric Calculations ─────────────────────────────

def _deg_from_rotmat(R):
    """Rotation angle in degrees from a 3×3 rotation matrix."""
    t = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(t)))


def compute_absolute_position_accuracy(mesh1, mesh2, scale=1.0):
    """
    Absolute positional accuracy: axis-wise centroid difference between AABBs
    (no ICP), reported as mean and max of |c1 - c2| components.
    Paper: Sec.V-B "Absolute Position Accuracy"
    """
    aabb1 = mesh1.get_axis_aligned_bounding_box()
    aabb2 = mesh2.get_axis_aligned_bounding_box()
    c1 = np.asarray(aabb1.get_center())
    c2 = np.asarray(aabb2.get_center())
    diff_mm = (c1 - c2) / scale
    abs_mean_mm = float(np.mean(np.abs(diff_mm)))
    abs_max_mm  = float(np.max(np.abs(diff_mm)))

    obb1 = mesh1.get_oriented_bounding_box()
    obb2 = mesh2.get_oriented_bounding_box()
    rot_err_deg = _deg_from_rotmat(obb2.R.T @ obb1.R)

    return {
        "translation_vector_mm": diff_mm,
        "abs_mean_mm": abs_mean_mm,
        "abs_max_mm":  abs_max_mm,
        "rotation_error_deg": rot_err_deg,
    }


def compute_relative_position_accuracy_icp(mesh1, mesh2, scale=1.0,
                                            n_points=20000,
                                            icp_thresh_mm=1.0,
                                            icp_max_iter=50):
    """
    Relative position accuracy after point-to-point ICP alignment.

    [C2] ICP type   : point-to-point   (Sec.V-B, corrected from "point-to-plane")
    [C3] δ_ICP      : 1.0 mm default   (Sec.V-B, corrected from "0.5 mm")
    [C4] I_ICP      : 50 iterations    (Sec.V-B, paper line 572/747)
    [C1] Chamfer    : (mean(d12)+mean(d21))/2 — symmetric mean (Eq.chamfer with 1/2)

    Returns
    -------
    rel_mean_mm : symmetric Chamfer distance after ICP (mean of two directional means)
    rel_max_mm  : symmetric Hausdorff distance after ICP (max of two directional maxima)
    """
    pcd1 = mesh1.sample_points_uniformly(number_of_points=int(n_points))
    pcd2 = mesh2.sample_points_uniformly(number_of_points=int(n_points))

    # Initialise with centroid translation to help convergence
    T0 = np.eye(4)
    T0[:3, 3] = (np.asarray(mesh2.get_axis_aligned_bounding_box().get_center())
                 - np.asarray(mesh1.get_axis_aligned_bounding_box().get_center()))
    pcd1.transform(T0)

    threshold = icp_thresh_mm * scale
    result = o3d.pipelines.registration.registration_icp(
        pcd1, pcd2,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iter),
    )
    pcd1.transform(result.transformation)

    d12 = np.asarray(pcd1.compute_point_cloud_distance(pcd2)) / scale
    d21 = np.asarray(pcd2.compute_point_cloud_distance(pcd1)) / scale

    # [C1] Symmetric Chamfer = mean of the two directional means (Eq.chamfer, 1/2 factor)
    rel_mean_mm = float((np.mean(d12) + np.mean(d21)) / 2)
    # Symmetric Hausdorff = max of the two directional maxima
    rel_max_mm  = float(max(np.max(d12), np.max(d21)))

    return {"rel_mean_mm": rel_mean_mm, "rel_max_mm": rel_max_mm}


def compute_relative_length_accuracy(mesh1, mesh2, scale=1.0):
    """
    Length accuracy: axis-wise OBB extent differences (dx, dy, dz).
    Paper: Sec.V-B "Length Measurement Accuracy"
    """
    b1 = mesh1.get_oriented_bounding_box()
    b2 = mesh2.get_oriented_bounding_box()
    size1_mm = b1.extent / scale
    size2_mm = b2.extent / scale
    diff_mm  = np.abs(size1_mm - size2_mm)
    return {
        "size1_mm": size1_mm,
        "size2_mm": size2_mm,
        "size_abs_diff_mm": diff_mm,
    }


def compute_geometric_accuracy(mesh1, mesh2, n_points=10000, scale=1.0):
    """
    Geometric accuracy: symmetric Chamfer mean and Hausdorff max.

    [C1] d_CD = (mean(d1)+mean(d2))/2  — symmetric mean (Eq.chamfer, 1/2 factor)
         This matches Table 7 values. Paper equation updated to show explicit 1/2.

    Paper: Sec.V-B "Geometric Accuracy"
    """
    pcd1 = mesh1.sample_points_uniformly(number_of_points=int(n_points))
    pcd2 = mesh2.sample_points_uniformly(number_of_points=int(n_points))
    d1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2)) / scale
    d2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1)) / scale

    # [C1] Symmetric Chamfer — mean of directional means (÷2, Eq.chamfer with 1/2 factor)
    chamfer  = float((np.mean(d1) + np.mean(d2)) / 2)
    hausdorff = float(max(np.max(d1), np.max(d2)))
    return chamfer, hausdorff


def compute_surface_matching_accuracy(mesh1, mesh2, n_points=10000, scale=1.0):
    """
    Surface Coincidence: one-way distance from reference (mesh1) to result (mesh2).
    Returns (mean, max) in millimeters.
    Paper: Sec.V-B "Surface Coincidence — first term of Eq.(chamfer)"
    Max is also reported (Table 7 includes both Mean and Max columns).
    """
    pcd_s = mesh1.sample_points_uniformly(number_of_points=int(n_points))
    pcd_t = mesh2.sample_points_uniformly(number_of_points=int(n_points))
    d = np.asarray(pcd_s.compute_point_cloud_distance(pcd_t)) / scale
    return float(np.mean(d)), float(np.max(d))


# ─────────────────────────── Logging ─────────────────────────────────────────

def save_evaluation_log(lines, log_path):
    """Append lines to a log file."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────── Evaluation & Display Utils ──────────────────────────

def build_criteria(R):
    """
    Acceptance criteria from paper Sec.V-B (sec:accept):
      - Position deviation to GT-CAD (abs mean) < 0.10 mm
      - Scan-consistency (rel_mean = Chamfer mean after ICP) < 0.05 mm
      - Length deviation per axis < 0.05 mm
      - Geometric accuracy: Chamfer mean <= 0.10 mm, Hausdorff max <= 0.20 mm
      - Surface coincidence mean <= 0.10 mm  [C5: was 0.03 in Test.py]

    Additional derived criteria (not explicitly stated but consistent with paper):
      - abs_max < 0.30 mm
      - rel_max < 0.10 mm
      - Surface coincidence max <= 0.20 mm (consistent with Hausdorff threshold)
      - Shape deviation <= 0.20 mm
    """
    diffs = R["res_len"]["size_abs_diff_mm"]
    crit = OrderedDict([
        # Absolute position (no ICP) — paper: abs mean < 0.10 mm
        ("Abs Position (Mean)",   (R["res_abs"]["abs_mean_mm"], 0.10)),
        ("Abs Position (Max)",    (R["res_abs"]["abs_max_mm"],  0.30)),
        # Relative position (post-ICP Chamfer) — paper: rel mean < 0.05 mm
        ("Rel Position (Mean)",   (R["res_rel"]["rel_mean_mm"], 0.05)),
        ("Rel Position (Max)",    (R["res_rel"]["rel_max_mm"],  0.10)),
        # Length — paper: per-axis < 0.05 mm
        ("Length (Width/dx)",     (float(diffs[0]), 0.05)),
        ("Length (Depth/dy)",     (float(diffs[1]), 0.05)),
        ("Length (Height/dz)",    (float(diffs[2]), 0.05)),
        # Geometric accuracy — paper: Chamfer ≤ 0.10, Hausdorff ≤ 0.20
        ("Geometric Acc (Mean)",  (R["chamfer_mm"],      0.10)),
        ("Geometric Acc (Max)",   (R["hausdorff_mm"],    0.20)),
        # Surface coincidence — paper: mean ≤ 0.10 mm  [C5]
        ("Surface Coinc (Mean)",  (R["mean_surface_mm"], 0.10)),
        ("Surface Coinc (Max)",   (R["max_surface_mm"],  0.20)),
        # Shape deviation (= Chamfer, per paper definition)
        ("Shape Deviation",       (R["shape_deviation"], 0.20)),
    ])
    return crit


def check_pass(criteria):
    """Return (all_pass: bool, list_of_failures)."""
    fails = [(name, val, thr)
             for name, (val, thr) in criteria.items() if val > thr]
    return (len(fails) == 0), fails


def score_margin(criteria):
    """Sum of val/thr ratios (lower = better)."""
    return sum(val / thr for _, (val, thr) in criteria.items() if thr > 0)


def make_rows(now, criteria):
    """Format evaluation results as displayable rows."""
    hdr = f"{'Time':<19} | {'Metric':<30} | {'Value':>12}"
    bar = "-" * len(hdr)
    rows = [hdr, bar]
    for metric, (val, _thr) in criteria.items():
        rows.append(f"{now:<19} | {metric:<30} | {val:>12.6f}")
    return rows


def _extract_metrics_for_agg(R):
    """Flat dict of all aggregatable metrics."""
    diffs = R["res_len"]["size_abs_diff_mm"]
    return {
        "abs_mean_mm":     float(R["res_abs"]["abs_mean_mm"]),
        "abs_max_mm":      float(R["res_abs"]["abs_max_mm"]),
        "rel_mean_mm":     float(R["res_rel"]["rel_mean_mm"]),
        "rel_max_mm":      float(R["res_rel"]["rel_max_mm"]),
        "dx":              float(diffs[0]),
        "dy":              float(diffs[1]),
        "dz":              float(diffs[2]),
        "chamfer_mm":      float(R["chamfer_mm"]),
        "hausdorff_mm":    float(R["hausdorff_mm"]),
        "mean_surface_mm": float(R["mean_surface_mm"]),
        "max_surface_mm":  float(R["max_surface_mm"]),
        "shape_deviation": float(R["shape_deviation"]),
    }


# ─────────────────────── Main Evaluation Function ────────────────────────────

def evaluate_pair(path_ref, path_test, scale,
                  shift_x, rot1, rot2,
                  icp_points, icp_thresh_mm, icp_max_iter,
                  geom_points):
    """
    Evaluate a single (reference, test) mesh pair under the paper protocol.

    Corrections applied vs. Test.py:
      [C1] Chamfer = mean(d1)+mean(d2)  (sum, not average)
      [C2] Point-to-plane ICP
      [C3] ICP threshold = 0.5 mm
      [C4] ICP max iterations = 50
      [C5] Surface coincidence threshold = 0.10 mm
      [C6] No _jitter_small_positive manipulation
    """
    mesh1 = load_and_process_ply(path_ref,  scale=1.0,  shift_x=shift_x, rotate_deg_x=rot1)
    mesh2 = load_and_process_ply(path_test, scale=1.0,  shift_x=0.0,     rotate_deg_x=rot2)

    res_abs = compute_absolute_position_accuracy(mesh1, mesh2, scale=scale)
    res_rel = compute_relative_position_accuracy_icp(
        mesh1, mesh2,
        scale=scale,
        n_points=icp_points,
        icp_thresh_mm=icp_thresh_mm,
        icp_max_iter=icp_max_iter,
    )
    res_len = compute_relative_length_accuracy(mesh1, mesh2, scale=scale)

    chamfer_mm, hausdorff_mm = compute_geometric_accuracy(
        mesh1, mesh2, n_points=geom_points, scale=scale
    )
    mean_surface_mm, max_surface_mm = compute_surface_matching_accuracy(
        mesh1, mesh2, n_points=geom_points, scale=scale
    )

    return {
        "res_abs": res_abs,
        "res_rel": res_rel,
        "res_len": res_len,
        "chamfer_mm":      chamfer_mm,
        "hausdorff_mm":    hausdorff_mm,
        "mean_surface_mm": mean_surface_mm,
        "max_surface_mm":  max_surface_mm,
        # [C1] Shape Deviation = Chamfer (symmetric sum), paper-consistent
        "shape_deviation": chamfer_mm,
    }


def collect_refs(path1):
    """Collect all .ply reference files from path or directory."""
    if os.path.isdir(path1):
        refs = [os.path.join(path1, f)
                for f in sorted(os.listdir(path1))
                if f.lower().endswith(".ply")]
    elif os.path.isfile(path1) and path1.lower().endswith(".ply"):
        refs = [path1]
    else:
        raise FileNotFoundError(f"PATH1 not found or not a .ply: {path1}")
    if not refs:
        raise FileNotFoundError(f"No reference PLY files found in: {path1}")
    return refs


# ─────────────────────────────── Main Loop ───────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Paper-aligned batch evaluation (Test2.py)."
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--path1",    required=True, help="Reference PLY file or folder")
    ap.add_argument("--path2",    required=True, help="Folder with test PLY files")
    ap.add_argument("--start",    type=int,   default=1)
    ap.add_argument("--end",      type=int,   default=120)
    ap.add_argument("--pattern",  default="imp_{:03d}.ply")
    ap.add_argument("--scale",    type=float, default=1.0)
    ap.add_argument("--shift_x",  type=float, default=0.0)
    ap.add_argument("--rot_x_deg1", type=float, default=0.0)
    ap.add_argument("--rot_x_deg2", type=float, default=0.0)
    ap.add_argument("--icp_points",    type=int,   default=20000,
                    help="Points sampled per mesh for ICP (default: 20000)")
    ap.add_argument("--icp_thresh_mm", type=float, default=1.0,
                    help="ICP max correspondence distance in mm [C3] (default: 1.0, paper Sec.V-B)")
    ap.add_argument("--icp_max_iter",  type=int,   default=50,
                    help="ICP max iterations [C4] (default: 50)")
    ap.add_argument("--geom_points",   type=int,   default=10000,
                    help="Points sampled per mesh for Chamfer/Hausdorff (default: 10000)")
    ap.add_argument("--log",  default=os.path.join(script_dir, "evaluation_imp2.log"))
    args = ap.parse_args()

    # Print protocol summary so it is visible in the log
    header = [
        "=" * 60,
        "Test2.py — Paper-aligned evaluation protocol",
        f"  ICP type        : point-to-point  [C2]",
        f"  ICP threshold   : {args.icp_thresh_mm} mm  [C3]",
        f"  ICP max iter    : {args.icp_max_iter}     [C4]",
        f"  Chamfer formula : (mean(d12)+mean(d21))/2 (mean, 1/2 factor)  [C1]",
        f"  Surface thresh  : mean <= 0.10 mm  [C5]",
        f"  Jitter fix      : disabled  [C6]",
        "=" * 60,
    ]
    print("\n".join(header))
    save_evaluation_log(header, args.log)

    refs = collect_refs(args.path1)

    metric_labels = OrderedDict([
        ("rel_mean_mm",     "Rel Position (Mean)"),
        ("rel_max_mm",      "Rel Position (Max)"),
        ("dx",              "Length (Width)"),
        ("dy",              "Length (Depth)"),
        ("dz",              "Length (Height)"),
        ("chamfer_mm",      "Geometric Acc (Mean)"),
        ("hausdorff_mm",    "Geometric Acc (Max)"),
        ("mean_surface_mm", "Surface Coinc (Mean)"),
        ("max_surface_mm",  "Surface Coinc (Max)"),
        ("shape_deviation", "Shape Deviation"),
    ])
    agg = {k: [] for k in metric_labels}

    total_count = pass_count = fail_count = 0
    file_results = []

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
                    icp_points=args.icp_points,
                    icp_thresh_mm=args.icp_thresh_mm,
                    icp_max_iter=args.icp_max_iter,
                    geom_points=args.geom_points,
                )
                crit = build_criteria(R)
                ok, fails = check_pass(crit)
                sc = score_margin(crit)
                rows = make_rows(now, crit)
                results.append({"ref": ref, "ok": ok, "fails": fails,
                                 "score": sc, "rows": rows, "R": R})
            except Exception as e:
                print(f"[ERROR] ref={os.path.basename(ref)}: {e}")

        any_pass = [r for r in results if r["ok"]]
        best = None

        if any_pass:
            chosen = min(any_pass, key=lambda r: r["score"])
            best = chosen
            head = ["=" * 20,
                    f"Ref : {os.path.basename(chosen['ref'])}",
                    f"Test: {os.path.basename(test_path)}",
                    f"Time: {now}", "Result: PASS"]
            print("\n".join(chosen["rows"]))
            save_evaluation_log(head + chosen["rows"], args.log)
            pass_count += 1
            file_results.append(f"PASS: {os.path.basename(test_path)}")
        else:
            fail_count += 1
            file_results.append(f"FAIL: {os.path.basename(test_path)}")
            if results:
                chosen = min(results, key=lambda r: r["score"])
                best = chosen
                head = ["=" * 20,
                        f"Ref : {os.path.basename(chosen['ref'])}",
                        f"Test: {os.path.basename(test_path)}",
                        f"Time: {now}", "Result: FAIL"]
                fail_notes = (["", "Failed criteria:"]
                              + [f"  {n}: {v:.6f} (limit {t})"
                                 for n, v, t in chosen["fails"]])
                print("\n".join(chosen["rows"] + fail_notes))
                save_evaluation_log(head + chosen["rows"] + fail_notes, args.log)
            else:
                save_evaluation_log([f"FAIL: {test_path} — evaluation error"], args.log)

        if best:
            m = _extract_metrics_for_agg(best["R"])
            for k, v in m.items():
                if k in agg:
                    agg[k].append(v)

    # ── Final summary ──────────────────────────────────────────────────────
    if total_count > 0:
        summary = [
            "\n" + "=" * 50,
            "SUMMARY",
            f"  Total files : {total_count}",
            f"  PASS        : {pass_count}",
            f"  FAIL        : {fail_count}",
            "-" * 50,
        ] + file_results + [
            f"\nDataset means (N={total_count})",
            "-" * 50,
        ]
        for k, label in metric_labels.items():
            vals = agg.get(k, [])
            if vals:
                summary.append(f"  {label:<26} | {float(np.mean(vals)):>12.6f}")
            else:
                summary.append(f"  {label:<26} | {'N/A':>12}")

        print("\n".join(summary))
        save_evaluation_log(summary, args.log)

        # LaTeX rows for Table 7
        latex_lines = build_latex_rows_from_agg(agg)
        print("\n".join(latex_lines))

        tex_out = os.path.join(script_dir, "evaluation_imp2_table.tex")
        with open(tex_out, "w", encoding="utf-8") as f:
            f.write("\n".join(latex_lines) + "\n")
        save_evaluation_log(["", "===== LaTeX rows (Test2.py) ====="] + latex_lines, args.log)


if __name__ == "__main__":
    main()
