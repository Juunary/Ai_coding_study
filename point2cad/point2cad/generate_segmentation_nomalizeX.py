import torch
import argparse
import numpy as np
from src.PointNet import PrimitivesEmbeddingDGCNGn
from src.fitting_utils import pca_numpy
from src.mean_shift import MeanShift
from src.segment_utils import rotation_matrix_a_to_b


def guard_mean_shift(embedding, quantile, iterations, kernel_type="gaussian"):
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="ParseNet Segmentation Prediction")
    parser.add_argument("--path_in", type=str, default="./assets/abc_00323.xyz")
    parser.add_argument("--with_normals", type=bool, default=False)
    cfg = parser.parse_args()

    num_channels = 6 if cfg.with_normals else 3
    pth_path = "./logs/pretrained_models/parsenet.pth" if cfg.with_normals else "./logs/pretrained_models/parsenet_no_normals.pth"

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
    model.load_state_dict(torch.load(pth_path, map_location=device))

    iterations = 50
    quantile = 0.015

    # ✅ 원본 좌표 불러오기 (normalize 제거)
    original_points = np.loadtxt(cfg.path_in).astype(np.float32)

    # ✅ 바로 Tensor로 변환
    points_tensor = torch.from_numpy(original_points)[None, :].to(device)

    with torch.no_grad():
        embedding, _, _ = model(
            points_tensor.permute(0, 2, 1), 
            torch.zeros_like(points_tensor)[:, 0], 
            False
        )
    embedding = torch.nn.functional.normalize(embedding[0].T, p=2, dim=1)

    _, _, cluster_ids = guard_mean_shift(
        embedding, quantile, iterations, kernel_type="gaussian"
    )

    cluster_ids = cluster_ids.data.cpu().numpy()

    # ✅ 결과는 반드시 원본 좌표로 저장
    np.savetxt(
        cfg.path_in.replace(".xyz", "_prediction.xyzc"),
        np.hstack([original_points, cluster_ids[:, None]]),
    )
