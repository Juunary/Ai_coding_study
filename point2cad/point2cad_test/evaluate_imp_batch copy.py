#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 예)
# python evaluate_imp_batch.py --path1 "/mnt/nas4/junewookang/point2cad/assets/impeller/ply_validation"   --path2 "./mnt/nas4/junewookang/point2cad/assets/impeller/ply_test"  --start 1 --end 120 --pattern "imp_{:03d}.ply"    --scale 1 --log "evaluation_imp2.log"
# python evaluate_imp_batch.py   --path1 "/mnt/nas4/junewookang/point2cad/assets/impeller/ply_validation"   --path2 "/mnt/nas4/junewookang/point2cad/assets/impeller/ply_test"   --start 1 --end 120   --pattern "imp_{:03d}.ply"   --scale 1   --log "evaluation_imp2.log"
import os
import argparse
import numpy as np
import open3d as o3d
from datetime import datetime
from collections import OrderedDict
from scipy.spatial import cKDTree
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

# ----------------------- 단면(특정 특징) 유틸 -----------------------
def _slice_mask_x(points_xyz, x0_mm, thickness_mm, scale=1.0):
    x0_internal = x0_mm * scale
    half = (thickness_mm * scale) / 2.0
    x = points_xyz[:, 0]
    return np.abs(x - x0_internal) <= half

def _project_yz_mm(points_xyz, scale=1.0):
    return points_xyz[:, 1:3] / scale  # mm 단위

def _convex_hull_area_2d(pts):  # scipy 없는 경우 대비(선택적)
    try:
        from scipy.spatial import ConvexHull
        if len(pts) >= 3:
            return float(ConvexHull(pts).volume)  # 2D에서 volume=area
        return 0.0
    except Exception:
        return 0.0

def _nearest_dists_A_to_B(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A 각 점에서 B까지 최근접 거리(유클리드)만 계산 — 메모리 안전."""
    if len(A) == 0:
        return np.empty((0,), dtype=np.float64)
    if len(B) == 0:
        return np.full((len(A),), np.inf, dtype=np.float64)
    treeB = cKDTree(B)
    dists, _ = treeB.query(A, k=1, workers=-1)
    return dists.astype(np.float64, copy=False)

def _hausdorff_2d(A, B):
    dA = _nearest_dists_A_to_B(A, B)
    dB = _nearest_dists_A_to_B(B, A)
    return float(max( (dA.max() if dA.size else 0.0),
                      (dB.max() if dB.size else 0.0) ))

def _chamfer_2d(A, B):
    dA = _nearest_dists_A_to_B(A, B)
    dB = _nearest_dists_A_to_B(B, A)
    mA = (dA.mean() if np.isfinite(dA).any() else np.inf)
    mB = (dB.mean() if np.isfinite(dB).any() else np.inf)
    return float((mA + mB) / 2.0)

def _centroid_dist_2d(A, B):
    c1 = np.mean(A, axis=0) if len(A) else np.array([0.0, 0.0])
    c2 = np.mean(B, axis=0) if len(B) else np.array([0.0, 0.0])
    return float(np.linalg.norm(c1 - c2))

def _bbox_sim_2d(A, B):
    def dims(P):
        if len(P) == 0:
            return np.array([0.0, 0.0])
        return np.max(P, axis=0) - np.min(P, axis=0)
    d1, d2 = dims(A), dims(B)
    sims = []
    for a, b in zip(d1, d2):
        if a == 0 and b == 0: sims.append(1.0)
        elif a == 0 or b == 0: sims.append(0.0)
        else: sims.append(min(a, b) / max(a, b))
    return float(np.mean(sims)) if sims else 0.0

def _density_sim_2d(A, B, grid_size=50):
    if len(A) == 0 or len(B) == 0:
        return 0.0
    allp = np.vstack([A, B])
    y_min, y_max = np.min(allp[:,0]), np.max(allp[:,0])
    z_min, z_max = np.min(allp[:,1]), np.max(allp[:,1])
    if y_max == y_min or z_max == z_min:
        return 0.0
    y_edges = np.linspace(y_min, y_max, grid_size + 1)
    z_edges = np.linspace(z_min, z_max, grid_size + 1)
    h1, _, _ = np.histogram2d(A[:,0], A[:,1], bins=[y_edges, z_edges])
    h2, _, _ = np.histogram2d(B[:,0], B[:,1], bins=[y_edges, z_edges])
    if h1.sum() > 0: h1 = h1 / h1.sum()
    if h2.sum() > 0: h2 = h2 / h2.sum()
    v1, v2 = h1.flatten(), h2.flatten()
    dot = float(np.dot(v1, v2))
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    return float(dot / (n1 * n2)) if (n1 > 0 and n2 > 0) else 0.0

def _improved_mean_error_2d(A, B):
    """상위 5% 아웃라이어 제거 + 양방향 평균/가중/중앙값 중 최솟값(KD-Tree 기반)."""
    dA = _nearest_dists_A_to_B(A, B)  # A→B
    dB = _nearest_dists_A_to_B(B, A)  # B→A

    dA = dA[np.isfinite(dA)]
    dB = dB[np.isfinite(dB)]
    if dA.size == 0 or dB.size == 0:
        return np.inf

    p95_A = np.percentile(dA, 95)
    p95_B = np.percentile(dB, 95)
    fA = dA[dA <= p95_A] if dA.size else dA
    fB = dB[dB <= p95_B] if dB.size else dB
    if fA.size == 0: fA = dA
    if fB.size == 0: fB = dB

    mean_avg = (fA.mean() + fB.mean()) / 2.0
    total = max(len(A) + len(B), 1)
    wA, wB = len(A)/total, len(B)/total
    wmean = wA * fB.mean() + wB * fA.mean()
    med = np.median(np.concatenate([fA, fB]))
    return float(min(mean_avg, wmean, med))

def compute_section_metrics(mesh1, mesh2, x_plane_mm=0.0, thickness_mm=0.5,
                            n_sample=300_000, scale=1.0, tol_overlap_mm=1e-7,
                            max_slice_points=120000, voxel_mm=0.0):
    """
    메쉬 → 균일 샘플 → x=상수 슬라이스 → YZ(mm) 투영 → 2D 유사도/오차 계산
    반환 dict의 단위는 모두 mm 기반(이미 scale 보정됨)
    """
    p1 = mesh1.sample_points_uniformly(number_of_points=int(n_sample))
    p2 = mesh2.sample_points_uniformly(number_of_points=int(n_sample))
    P1 = np.asarray(p1.points); P2 = np.asarray(p2.points)

    m1 = _slice_mask_x(P1, x_plane_mm, thickness_mm, scale)
    m2 = _slice_mask_x(P2, x_plane_mm, thickness_mm, scale)
    S1 = _project_yz_mm(P1[m1], scale)
    S2 = _project_yz_mm(P2[m2], scale)

    # 2D 보셀 다운샘플(선택)
    if voxel_mm and voxel_mm > 0.0:
        def voxel_ds(pts, vox):
            if len(pts) == 0: return pts
            key = np.floor(pts / vox).astype(np.int64)
            # 각 보셀 첫 번째 포인트만 채택
            _, idx = np.unique(key, axis=0, return_index=True)
            return pts[np.sort(idx)]
        S1 = voxel_ds(S1, voxel_mm)
        S2 = voxel_ds(S2, voxel_mm)

    # 포인트 수 상한
    rng = np.random.default_rng(0)
    if len(S1) > max_slice_points:
        idx = rng.choice(len(S1), size=max_slice_points, replace=False)
        S1 = S1[idx]
    if len(S2) > max_slice_points:
        idx = rng.choice(len(S2), size=max_slice_points, replace=False)
        S2 = S2[idx]

    if len(S1) == 0 or len(S2) == 0:
        return {
            "sec_valid": False,
            "sec_count1": int(len(S1)),
            "sec_count2": int(len(S2))
        }

    # 이하 동일(단, haus/cham/imp_mean은 위 KD-Tree 버전 사용)
    haus = _hausdorff_2d(S1, S2)
    cham = _chamfer_2d(S1, S2)
    area1, area2 = _convex_hull_area_2d(S1), _convex_hull_area_2d(S2)
    area_sim = (min(area1, area2) / max(area1, area2)) if max(area1, area2) > 0 else (1.0 if area1==area2==0 else 0.0)
    cdist = _centroid_dist_2d(S1, S2)
    bbox_sim = _bbox_sim_2d(S1, S2)
    dens_sim = _density_sim_2d(S1, S2)
    imp_mean = _improved_mean_error_2d(S1, S2)

    tree1 = cKDTree(S1)
    d21, _ = tree1.query(S2, k=1, workers=-1)
    overlap_rate = float((d21 <= tol_overlap_mm).sum() / len(S2) * 100.0)

    max_expected = 50.0  # mm
    haus_sim = max(0.0, 1.0 - haus/max_expected)
    cham_sim = max(0.0, 1.0 - cham/max_expected)
    cen_sim  = max(0.0, 1.0 - cdist/max_expected)
    weights = dict(haus=0.2, cham=0.3, area=0.15, cen=0.1, bbox=0.15, dens=0.1)
    overall = (weights['haus']*haus_sim + weights['cham']*cham_sim +
               weights['area']*area_sim + weights['cen']*cen_sim +
               weights['bbox']*bbox_sim + weights['dens']*dens_sim)

    return {
        "sec_valid": True,
        "sec_count1": int(len(S1)),
        "sec_count2": int(len(S2)),
        "sec_hausdorff_mm": float(haus),
        "sec_chamfer_mm": float(cham),
        "sec_area1_mm2": float(area1),
        "sec_area2_mm2": float(area2),
        "sec_area_sim": float(area_sim),
        "sec_centroid_dist_mm": float(cdist),
        "sec_bbox_sim": float(bbox_sim),
        "sec_density_sim": float(dens_sim),
        "sec_improved_mean_mm": float(imp_mean),
        "sec_overlap_rate_pct": float(overlap_rate),
        "sec_overall_sim": float(overall)
    }

# ----------------------- 로깅 -----------------------
def save_evaluation_log(lines, log_path):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# ----------------------- 평가/표출 유틸 -----------------------
def build_criteria(R, sec=None):
    """
    순서 고정. 값은 '작을수록 좋은' 값으로 맞춰서 배치.
    유사도(0~1, 1이 좋음)는 (1-유사도)로 변환하여 임계와 비교.
    """
    crit = OrderedDict([
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
    # --- 추가: 특정특징(단면) — 다른 지표와 '같은 방식'으로 1줄만
    if R.get("sec_feature_mm") is not None:
        crit.update([
            ("특정특징 정확도", (R["sec_feature_mm"], R.get("sec_tol_mm_val", 0.05))),
        ])
    return crit

def check_pass(criteria):
    fails = [(name, val, thr) for name, (val, thr) in criteria.items() if val > thr]
    return (len(fails) == 0), fails

def score_margin(criteria):
    # 작은 값이 더 좋음 (임계 대비 비율의 합)
    return sum((val / thr) for (_, (val, thr)) in criteria.items())
 

def make_rows(now, criteria, sec=None, R=None):
    hdr = f"{'시간':<19} | {'측정 대상':<30} | {'측정 값':>12}"
    bar = "-" * len(hdr)
    rows = [hdr, bar]
    for metric, (val, _thr) in criteria.items():
        rows.append(f"{now:<19} | {metric:<30} | {val:>12.6f}")
    return rows
def _extract_metrics_for_agg(R):
    dx = float(abs(R["res_len"]["size1_mm"][0] - R["res_len"]["size2_mm"][0]))
    dy = float(abs(R["res_len"]["size1_mm"][1] - R["res_len"]["size2_mm"][1]))
    dz = float(abs(R["res_len"]["size1_mm"][2] - R["res_len"]["size2_mm"][2]))
    return {
        "abs_mean_mm":      float(R["res_abs"]["abs_mean_mm"]),
        "abs_max_mm":       float(R["res_abs"]["abs_max_mm"]),
        "rel_mean_mm":      float(R["res_rel"]["rel_mean_mm"]),
        "rel_max_mm":       float(R["res_rel"]["rel_max_mm"]),
        "dx":               dx,
        "dy":               dy,
        "dz":               dz,
        "chamfer_mm":       float(R["chamfer_mm"]),
        "hausdorff_mm":     float(R["hausdorff_mm"]),
        "mean_surface_mm":  float(R["mean_surface_mm"]),
        "max_surface_mm":   float(R["max_surface_mm"]),
        "vol_err_percent":  float(R["res_vol"]["err_percent"]),
        # 요구 사항: 특정특징(sec_feature_mm)은 평균에서 제외
    }
# ----------------------- 평가 함수 -----------------------
def evaluate_pair(path_ref, path_test, scale, shift_x, rot1, rot2,
                  icp_points, icp_thresh_mm, geom_points,
                  sec_enable=True, sec_x=0.0, sec_thick=0.5,
                  sec_sample=300_000, sec_tol_overlap=1e-7,
                  sec_max_slice_points=120000, sec_voxel_mm=0.0, sec_tol_mm=0.05):
    mesh1 = load_and_process_ply(path_ref, scale=1.0, shift_x=shift_x, rotate_deg_x=rot1)
    mesh2 = load_and_process_ply(path_test, scale=1.0, shift_x=0.0,     rotate_deg_x=rot2)

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

    # 단면(특정 특징) 측정
    sec = None
    if sec_enable:
        sec = compute_section_metrics(
            mesh1, mesh2,
            x_plane_mm=sec_x,
            thickness_mm=sec_thick,
            n_sample=sec_sample,
            scale=scale,
            tol_overlap_mm=sec_tol_overlap,
            max_slice_points=sec_max_slice_points,
            voxel_mm=sec_voxel_mm
        )

    # --- 추가: 특정특징 스칼라(다른 지표와 동일한 처리에 쓰일 값)
    sec_feature_mm = None
    if sec and sec.get("sec_valid", False):
        # '같은 방식'을 위해 단일 수치만 기준표에 넣는다 (나쁜 쪽 기준)
        sec_feature_mm = float(max(sec["sec_chamfer_mm"], sec["sec_improved_mean_mm"]))

    return {
        "res_abs": res_abs, "res_rel": res_rel, "res_len": res_len,
        "chamfer_mm": chamfer_mm, "hausdorff_mm": hausdorff_mm,
        "mean_surface_mm": mean_surface_mm, "max_surface_mm": max_surface_mm,
        "res_vol": res_vol,
        "sec": sec,
        # ↓ 기준표와 동일한 방식으로 쓰기 위한 값/임계
        "sec_feature_mm": sec_feature_mm,
        "sec_tol_mm_val": sec_tol_mm,
    }

# ----------------------- 유틸: 기준 폴더에서 refs 수집 -----------------------
def collect_refs(path1):
    if os.path.isdir(path1):
        # 폴더인 경우: .ply 전부 수집
        refs = [os.path.join(path1, f) for f in sorted(os.listdir(path1))
                if f.lower().endswith(".ply")]
    elif os.path.isfile(path1) and path1.lower().endswith(".ply"):
        refs = [path1]
    else:
        raise FileNotFoundError(f"PATH1 not found or not a ply: {path1}")
    if len(refs) == 0:
        raise FileNotFoundError(f"기준 PLY가 없습니다: {path1}")
    return refs


# ----------------------- 메인 루프 -----------------------
def main():
    ap = argparse.ArgumentParser(description="Batch evaluate meshes using ALL validations in a folder (pass if ANY ref passes)")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_log_path = os.path.join(script_dir, "evaluation_imp.log")
    ap.add_argument("--path1", help="기준(원본) PLY 파일 경로 혹은 폴더", default="/mnt/nas4/junewookang/point2cad/assets/impeller/ply_validation")
    ap.add_argument("--path2", help="mesh_###.ply들이 있는 폴더", default="/mnt/nas4/junewookang/point2cad/assets/impeller/ply_test")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end",   type=int, default=120)
    ap.add_argument("--pattern", default="imp_{:03d}.ply")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--shift_x", type=float, default=0.0)
    ap.add_argument("--rot_x_deg1", type=float, default=0.0)  # 검증메쉬 회전(공통)
    ap.add_argument("--rot_x_deg2", type=float, default=0.0)  # 테스트메쉬 회전
    ap.add_argument("--icp_points", type=int, default=20000)
    ap.add_argument("--icp_thresh_mm", type=float, default=1.0)
    ap.add_argument("--geom_points", type=int, default=10000)
    ap.add_argument("--log", default=default_log_path)

    # 단면(특정 특징) 옵션
    ap.add_argument("--sec_enable", type=int, default=1)       # 1:on, 0:off
    ap.add_argument("--sec_x", type=float, default=0.0)        # x=0(mm)
    ap.add_argument("--sec_thick", type=float, default=0.5)    # ±thickness/2
    ap.add_argument("--sec_sample", type=int, default=300000)  # 표면 샘플 수
    ap.add_argument("--sec_tol_overlap", type=float, default=1e-7)
    ap.add_argument("--sec_max_slice_points", type=int, default=120000,  help="슬라이스 후 각 단면 최대 포인트 수(초과 시 랜덤 다운샘플)")
    ap.add_argument("--sec_voxel_mm",         type=float, default=0.0,   help="슬라이스 YZ 평면에서의 보셀 크기(mm). 0이면 미사용")
    ap.add_argument("--sec_tol_mm", type=float, default=0.05, help="특정특징(단면) 허용 오차(mm)")

    args = ap.parse_args()

    # 기준 목록 수집
    refs = collect_refs(args.path1)

    total_count = 0
    pass_count = 0
    fail_count = 0
    file_results = []

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
        ("vol_err_percent", "부피 오차(%)"),
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
                    sec_enable=bool(args.sec_enable),
                    sec_x=args.sec_x, sec_thick=args.sec_thick,
                    sec_sample=args.sec_sample,
                    sec_tol_overlap=args.sec_tol_overlap,
                    sec_max_slice_points=args.sec_max_slice_points,
                    sec_voxel_mm=args.sec_voxel_mm,
                    sec_tol_mm=args.sec_tol_mm,
                )
                crit = build_criteria(R, R.get("sec"))
                ok, fails = check_pass(crit)
                sc = score_margin(crit)
                rows = make_rows(now, crit, R.get("sec"), R)
                results.append({
                    "ref": ref, "crit": crit, "ok": ok, "fails": fails,
                    "score": sc, "rows": rows,
                    "sec_valid": bool(R.get("sec", {}).get("sec_valid", False)),
                    "R": R, 
                })
            except Exception as e:
                print(f"[ERROR] ref={os.path.basename(ref)}: {e}")

        any_pass = [r for r in results if r["ok"]]
        if any_pass:
            chosen = min(any_pass, key=lambda r: r["score"])
            head = [
                "====================",
                f"Mesh1(ref): {os.path.basename(chosen['ref'])}",
                f"Mesh2(test): {os.path.basename(test_path)}",
                f"평가 시각: {now}",
            ]
            special = "없음"
            if not chosen.get("sec_valid", True) and bool(args.sec_enable):
                special = "특정특징 단면 미검출(비유효)"
            tail = ["", f"특이사항: {special}"]

            out_lines = head + chosen["rows"] + tail

            print("\n".join(chosen["rows"]))
            print("\n".join(tail))
            save_evaluation_log(out_lines, args.log)
            pass_count += 1
            file_results.append(f"PASS: {os.path.basename(test_path)}")

            m = _extract_metrics_for_agg(chosen["R"])
            for k, v in m.items():
                agg[k].append(v)
        else:
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
                print("\n".join(r["rows"]))
                print("\n".join(notes))
            save_evaluation_log(all_lines, args.log)
            fail_count += 1
            file_results.append(f"FAIL: {os.path.basename(test_path)}")
            chosen = min(results, key=lambda r: r["score"])
            m = _extract_metrics_for_agg(chosen["R"])
            for k, v in m.items():
                agg[k].append(v)

    if total_count > 0:
        summary_lines = [
            "\n==================== SUMMARY ====================",
            f"총 평가 파일: {total_count}",
            f"통과(PASS): {pass_count}",
            f"실패(FAIL): {fail_count}",
            "-----------------------------------------------",
        ]
        # 파일별 PASS/FAIL 목록
        summary_lines.extend(file_results)

        # ===== 추가: 지표별 데이터셋 평균 =====
        summary_lines.append("\n----------- DATASET MEANS (각 지표: N={}) -----------".format(total_count))
        for k, label in metric_labels.items():
            vals = agg.get(k, [])
            if len(vals):
                mean_v = float(np.mean(vals))
                summary_lines.append(f"{label:<20} | {mean_v:>12.6f}")

        save_evaluation_log(summary_lines, args.log)

if __name__ == "__main__":
    main()
