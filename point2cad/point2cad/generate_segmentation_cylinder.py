import torch
import argparse
import numpy as np
from src.PointNet import PrimitivesEmbeddingDGCNGn
from src.fitting_utils import pca_numpy

# from src.segment_loss import EmbeddingLoss
from src.PointNet_seg import PointNetSeg               # ← 여기만 남기고
from src.mean_shift import MeanShift            # MeanShift 불필요
from src.fitting_utils import pca_numpy
from src.segment_utils import rotation_matrix_a_to_b
def guard_mean_shift(embedding, quantile, iterations, kernel_type="gaussian"):
    """
    Sometimes if bandwidth is small, number of cluster can be larger than 50,
    but we would like to keep max clusters 50 as it is the max number in our dataset.
    In that case you increase the quantile to increase the bandwidth to decrease
    the number of clusters.
    """
    ms = MeanShift()
    while True:
        _, center, bandwidth, cluster_ids = ms.mean_shift(
            embedding, 10000, quantile, iterations, kernel_type=kernel_type
        )
        if torch.unique(cluster_ids).shape[0] > 49:
            quantile *= 1.2
        else:
            break
    return center, bandwidth, cluster_ids


def normalize_points(points):
    EPS = np.finfo(np.float32).eps
    points = points - np.mean(points, 0, keepdims=True)
    S, U = pca_numpy(points)
    smallest_ev = U[:, np.argmin(S)]
    R = rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
    points = (R @ points.T).T
    std = np.max(points, 0) - np.min(points, 0)
    points = points / (np.max(std) + EPS)
    return points.astype(np.float32)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="PointNetSeg inference")
    parser.add_argument("--path_in", type=str, required=True)
    parser.add_argument("--model",   type=str, required=True)
    args = parser.parse_args()

    # ── ③ 모델 생성·로드 ───────────────────────────────
    model = PointNetSeg(num_classes=3).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # ── ④ 입력 로드 & 정규화 ──────────────────────────
    pts_raw = np.loadtxt(args.path_in).astype(np.float32)
    pts_norm = normalize_points(pts_raw)
    pts = torch.from_numpy(pts_norm)[None].to(device)  # (1,N,3)

    # ── ⑤ 추론 (argmax) ──────────────────────────────
    with torch.no_grad():
        logits = model(pts.permute(0,2,1))   # (1,3,N)
    pred = logits.argmax(1).cpu().numpy()[0]  # (N,)

    # ── ⑥ 저장 (.xyzc) ───────────────────────────────
    out_xyzc = np.hstack([pts_raw, pred[:,None]])
    out_path = args.path_in.replace(".xyz", "_Supervised_prediction.xyzc")
    np.savetxt(out_path, out_xyzc)
    print("✅ saved →", out_path)