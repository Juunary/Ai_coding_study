#!/usr/bin/env python3
# coding: utf-8
"""
Point2CAD ‒ segmentation inference script (XYZ / XYZ+N 입력 지원)

변경 사항
① 좌표‧법선 분리 → 좌표만 normalize → 필요하면 다시 결합
② (N×6) 배열에 normalize_points()를 두 번 적용하지 않음
③ 샘플링 로직을 함수로 분리해 가독성 향상
④ 출력 경로·파일명 생성 부분을 함수화


python point2cad/generate_segmentation_2.py --path_in './assets/xyz/imp_normals.xyz' --quantile 0.003 --with_normals True

quantile : 기본이 0.015인데 낮을수록 layer가 더 많이 나옵니다.
with_normals : 기본이 False인데 True로 하면 법선벡터를 이용해서 구합니다.
"""

import os
import argparse
import numpy as np
import torch

from src.PointNet import PrimitivesEmbeddingDGCNGn
from src.fitting_utils import pca_numpy
from src.mean_shift import MeanShift
from src.segment_utils import rotation_matrix_a_to_b

# ──────────────────────────────────────────────────────────────
# 유틸리티 함수
# ──────────────────────────────────────────────────────────────
def guard_mean_shift(embedding, quantile, iterations, kernel_type="gaussian"):
    """
    Mean-Shift를 수행하되, 군집 수가 50개를 초과하면
    bandwidth를 키우기 위해 quantile을 1.2배씩 늘려 재시도.
    """
    ms = MeanShift()
    while True:
        _, center, bandwidth, cluster_ids = ms.mean_shift(
            embedding, 10_000, quantile, iterations, kernel_type=kernel_type
        )
        if torch.unique(cluster_ids).shape[0] > 49:
            quantile *= 1.2
        else:
            break
    return center, bandwidth, cluster_ids


def normalize_points(points):
    """
    PCA 기반 정렬 + 최대 치수로 정규화 (N×3 전제)
    """
    EPS = np.finfo(np.float32).eps
    points = points - np.mean(points, 0, keepdims=True)
    S, U = pca_numpy(points)
    smallest_ev = U[:, np.argmin(S)]
    R = rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
    points = (R @ points.T).T
    extent = np.max(points, 0) - np.min(points, 0)
    points = points / (np.max(extent) + EPS)
    return points.astype(np.float32)


def random_subsample(arr, max_n):
    """
    행렬 arr(N×D)에서 중복 없이 max_n개 샘플링.
    N ≤ max_n 이면 그대로 반환.
    """
    n = arr.shape[0]
    if n > max_n:
        choice = np.random.choice(n, max_n, replace=False)
        return arr[choice], choice
    return arr, np.arange(n)


def build_save_path(path_in, quantile, out_dir):
    """
    입력 파일·quantile 값을 기반으로 출력 경로(확장자 .xyzc) 생성
    """
    stem = os.path.splitext(os.path.basename(path_in))[0]
    quantile_str = f"{quantile:.3f}"
    fname = f"{stem}_prediction_{quantile_str}.xyzc"
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, fname)


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(
        description="ParseNet segmentation prediction for Point2CAD"
    )
    parser.add_argument(
        "--path_in",
        type=str,
        default="./assets/xyz/cylinder.xyz",
        help="Input .xyz/xyzrgb/xyzrn file",
    )
    parser.add_argument("--with_normals", type=bool, default=False)
    parser.add_argument("--quantile", type=float, default=0.015)
    parser.add_argument("--max_points", type=int, default=30_000)
    parser.add_argument(
        "--save_dir", type=str, default="./assets/xyzc", help="Output folder"
    )
    cfg = parser.parse_args()

    # ── ① 좌표·법선 분리 & 좌표만 정규화 ──────────────────────────
    raw = np.loadtxt(cfg.path_in).astype(np.float32)
    coords = normalize_points(raw[:, :3])           # N×3

    if cfg.with_normals and raw.shape[1] >= 6:
        points = np.hstack([coords, raw[:, 3:6]])   # N×6 (XYZnormals)
        num_channels = 6
        pth_path = (
            "/mnt/nas4/lch/point2cad_main/point2cad/logs/pretrained_models/"
            "parsenet.pth"
        )
    else:
        points = coords                             # N×3
        num_channels = 3
        pth_path = (
            "/mnt/nas4/lch/point2cad_main/point2cad/logs/pretrained_models/"
            "parsenet_no_normals.pth"
        )

    # ── ② 필요 시 다운샘플링 ────────────────────────────────────
    points, idx_keep = random_subsample(points, cfg.max_points)
    print(
        f"[INFO] {raw.shape[0]:,} pts → "
        f"{'샘플링 ' if raw.shape[0] > cfg.max_points else ''}{points.shape[0]:,} pts"
    )

    # (coords만 두 번 normalize 하던 기존 오류 부분 삭제)  ③

    # ── PyTorch 텐서 변환 ──────────────────────────────────────
    pts_tensor = torch.from_numpy(points)[None].to(device)          # (1, N, D)

    # ── ④ 모델 로드 ────────────────────────────────────────────
    model = PrimitivesEmbeddingDGCNGn(
        embedding=True,
        emb_size=128,
        primitives=True,
        num_primitives=10,
        mode=0,
        num_channels=num_channels,
    )
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(pth_path))

    # ── 포워드 패스 & 임베딩 정규화 ─────────────────────────────
    with torch.no_grad():
        embedding, _, _ = model(pts_tensor.permute(0, 2, 1), torch.zeros(1).to(device), False)
    embedding = torch.nn.functional.normalize(embedding[0].T, p=2, dim=1)

    # ── Mean-Shift 클러스터링 ──────────────────────────────────
    _, _, cluster_ids = guard_mean_shift(
        embedding, cfg.quantile, iterations=50, kernel_type="gaussian"
    )
    cluster_ids = cluster_ids.cpu().numpy()  # (N,)

    # ── 결과 저장 (.xyzc: XYZ[+N] + cluster_id) ────────────────
    save_path = build_save_path(cfg.path_in, cfg.quantile, cfg.save_dir)
    # ▼ 여기 한 줄만 수정 ▼
    np.savetxt(save_path,
           np.hstack([points[:, :3],          # XYZ만 추출
                      cluster_ids[:, None]]), # + C
           fmt="%.6f")

    print(f"[OK] Saved: {save_path}")
