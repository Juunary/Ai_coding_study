#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XYZ 점군의 경계선 추출:
 - 국소 PCA로 곡률(= surface variation)과 주성분을 평가
 - 법선 급변(edge/corner) + 이웃 비대칭(경계) 스코어를 결합
 - 선택된 경계 점들로 KNN 그래프를 만들어 LineSet으로 저장/시각화

필요: open3d>=0.17, numpy, scipy
"""

import argparse
import os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def load_xyz(path):
    """XYZ 또는 XYZC 파일 로드 (x y z [c])"""
    arr = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            vals = line.strip().split()
            if len(vals) < 3:
                continue
            x, y, z = map(float, vals[:3])
            arr.append([x, y, z])
    pts = np.asarray(arr, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("입력 파일에서 점을 읽지 못했습니다.")
    return pts


def estimate_normals(pts, k_normal=30):
    """kNN로 법선 추정 (Open3D, 일관된 방향으로 정리)"""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_normal)
    )
    pcd.orient_normals_consistent_tangent_plane(k_normal)
    return np.asarray(pcd.normals), pcd


def local_pca_curvature(pts, k_curv=30):
    """
    PCA 고유값으로 surface variation(=λ3 / (λ1+λ2+λ3))을 곡률 근사치로.
    λ1 ≥ λ2 ≥ λ3 (3D 분산) 가정.
    """
    tree = cKDTree(pts)
    N = len(pts)
    curv = np.zeros(N, dtype=np.float64)
    eigvecs = np.zeros((N, 3, 3), dtype=np.float64)  # 주성분벡터(선택적 활용용)
    for i in range(N):
        _, idx = tree.query(pts[i], k=k_curv)
        nn = pts[idx]
        C = np.cov((nn - nn.mean(axis=0)).T)  # 3x3
        w, v = np.linalg.eigh(C)             # 오름차순: w0<=w1<=w2
        # 큰값부터 정렬
        order = np.argsort(w)[::-1]
        w = w[order]
        v = v[:, order]
        s = w.sum()
        curv[i] = (w[-1] / s) if s > 0 else 0.0  # surface variation ~ 곡률 근사
        eigvecs[i] = v
    return curv, eigvecs


def edge_scores(pts, normals, k_edge=30):
    """
    (1) 법선 급변도: 이웃과의 평균 법선 각도
    (2) 비대칭도: 이웃 점들의 (x_j - x_i)·n_i 부호 불균형(경계면 근처는 한쪽으로 편중)
    """
    tree = cKDTree(pts)
    N = len(pts)

    mean_normal_angle = np.zeros(N, dtype=np.float64)
    asymmetry = np.zeros(N, dtype=np.float64)

    for i in range(N):
        _, idx = tree.query(pts[i], k=k_edge)
        idx = np.atleast_1d(idx)
        nn = pts[idx]
        nn_normals = normals[idx]

        # (1) 법선 각도 평균 (라디안)
        ni = normals[i]
        dots = np.clip((nn_normals * ni).sum(axis=1), -1.0, 1.0)
        angles = np.arccos(dots)
        mean_normal_angle[i] = angles.mean()

        # (2) 비대칭도
        offset = nn - pts[i]
        proj = offset @ ni  # 이웃이 법선 방향 앞/뒤로 얼마나 있는지
        front = np.count_nonzero(proj > 0)
        back = np.count_nonzero(proj < 0)
        total = max(1, (front + back))
        asymmetry[i] = abs(front - back) / total  # 0~1, 1에 가까울수록 한쪽 편중
    return mean_normal_angle, asymmetry


def combine_scores(curv, mean_normal_angle, asymmetry,
                   w_curv=0.4, w_normal=0.4, w_asym=0.2):
    """
    점수 결합: 정규화 후 가중합
    - curv: surface variation (0~작은 값) → 0~1 정규화
    - mean_normal_angle: 0~π → /π
    - asymmetry: 0~1
    """
    eps = 1e-9
    c = (curv - curv.min()) / (curv.max() - curv.min() + eps)
    n = mean_normal_angle / np.pi
    a = asymmetry  # 이미 0~1

    score = w_curv * c + w_normal * n + w_asym * a
    return score, (c, n, a)


def knn_lines(indices, pts, k=6):
    """
    선택된 점(인덱스)들로 KNN 엣지를 만들어 LineSet 생성용 인덱스 쌍 반환.
    중복 제거를 위해 (min,max)로 정렬해 set으로 관리.
    """
    sub_pts = pts[indices]
    tree = cKDTree(sub_pts)
    pairs = set()
    for i in range(len(indices)):
        _, idx = tree.query(sub_pts[i], k=min(k, len(indices)))
        for j in idx:
            if i == j:
                continue
            a, b = sorted((i, j))
            pairs.add((a, b))
    # 전역 인덱스가 아니라 서브 인덱스 기준으로 반환
    return np.array(list(pairs), dtype=np.int32)


def main():
    ap = argparse.ArgumentParser(description="XYZ 점군 경계선 추출")
    ap.add_argument("--in_xyz", required=True, help="입력 .xyz 또는 .xyzc")
    ap.add_argument("--out_prefix", default="boundary",
                    help="출력 접두사 (PLY/XYZ 등)")
    ap.add_argument("--k_normal", type=int, default=30, help="법선용 k")
    ap.add_argument("--k_curv", type=int, default=30, help="곡률용 k")
    ap.add_argument("--k_edge", type=int, default=30, help="에지지표용 k")
    ap.add_argument("--top_p", type=float, default=0.60,
                    help="결합 스코어 상위 비율(0~1) 선택 (기본 20%)")
    ap.add_argument("--w_curv", type=float, default=0.4, help="곡률 가중치")
    ap.add_argument("--w_normal", type=float, default=0.4, help="법선각 가중치")
    ap.add_argument("--w_asym", type=float, default=0.2, help="비대칭 가중치")
    ap.add_argument("--knn_line", type=int, default=8, help="라인 그래프 K")
    ap.add_argument("--visualize", action="store_true", help="Open3D 시각화")
    args = ap.parse_args()

    pts = load_xyz(args.in_xyz)
    print(f"[INFO] points: {len(pts):,}")

    # 1) 법선
    normals, pcd = estimate_normals(pts, k_normal=args.k_normal)

    # 2) 곡률(=surface variation 근사)
    curv, _ = local_pca_curvature(pts, k_curv=args.k_curv)

    # 3) 에지 지표
    mean_normal_angle, asymmetry = edge_scores(
        pts, normals, k_edge=args.k_edge
    )

    # 4) 결합 스코어
    score, (c_norm, n_norm, a_norm) = combine_scores(
        curv, mean_normal_angle, asymmetry,
        w_curv=args.w_curv, w_normal=args.w_normal, w_asym=args.w_asym
    )

    # 상위 p% 선택
    thresh = np.quantile(score, 1.0 - args.top_p)
    sel_mask = score >= thresh
    sel_idx = np.where(sel_mask)[0]
    print(f"[INFO] selected boundary candidates: {len(sel_idx):,} "
          f"({args.top_p*100:.1f}% top by score, thresh={thresh:.4f})")

    # 저장: 선택점 XYZ
    out_xyz = f"{args.out_prefix}_points.xyz"
    np.savetxt(out_xyz, pts[sel_idx], fmt="%.6f")
    print(f"[SAVE] {out_xyz}")

    # 5) 선택점들로 라인 그래프 생성
    lines_local = knn_lines(sel_idx, pts, k=args.knn_line)
    # LineSet은 서브 인덱스 기준이므로, 좌표는 서브 포인트로 대입
    sub_pts = pts[sel_idx]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(sub_pts),
        lines=o3d.utility.Vector2iVector(lines_local),
    )

    # 색상(선: 진한 회색)
    colors = np.tile(np.array([[0.2, 0.2, 0.2]]), (len(lines_local), 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # 저장: PLY (LineSet은 .ply로 내보내면 edge list로 보존됨)
    out_ply = f"{args.out_prefix}_lines.ply"
    o3d.io.write_line_set(out_ply, line_set)
    print(f"[SAVE] {out_ply}")

    # 선택점 시각화용 PCD도 저장
    pcd_sel = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sub_pts))
    pcd_sel.paint_uniform_color([1.0, 0.1, 0.1])  # 빨강
    o3d.io.write_point_cloud(f"{args.out_prefix}_points.ply", pcd_sel)
    print(f"[SAVE] {args.out_prefix}_points.ply")

    if args.visualize:
        # 원본(연한 회색) + 경계점(빨강) + 라인
        base = pcd
        base.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([base, pcd_sel, line_set])


if __name__ == "__main__":
    main()
