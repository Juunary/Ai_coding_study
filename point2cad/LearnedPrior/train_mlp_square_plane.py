import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



# ✅ 1. 정사각형 평면 점 생성 함수 (xyz 파일 저장 포함)
def generate_square_plane_pointcloud(
    xyz_path, param_path, normal, offset,
    side_length=1.0, num_points=1000, noise=0.0, edge_ratio=0.3
):
    """
    edge_ratio: 가장자리에 배치할 점 비율 (예: 0.3 → 30%)
    """
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)

    # tangent 벡터 구하기
    if np.allclose(normal, [0, 0, 1]):
        tangent1 = np.array([1, 0, 0])
    else:
        tangent1 = np.cross(normal, [0, 0, 1])
        tangent1 = tangent1 / np.linalg.norm(tangent1)
    tangent2 = np.cross(normal, tangent1)

    # 점 분할 수
    num_edge = int(num_points * edge_ratio)
    num_inner = num_points - num_edge

    points = []

    # ▶ 내부 균등 점
    for _ in range(num_inner):
        u = np.random.uniform(-0.4, 0.4) * side_length
        v = np.random.uniform(-0.4, 0.4) * side_length
        p = u * tangent1 + v * tangent2 - offset * normal
        if noise > 0:
            p += np.random.normal(0, noise, size=3)
        points.append(p)

    # ▶ 가장자리 점 (테두리 쪽 u, v 중 하나는 ±0.5 고정)
    for _ in range(num_edge):
        edge_choice = np.random.choice(["u", "v"])
        if edge_choice == "u":
            u = np.random.choice([-0.5, 0.5]) * side_length
            v = np.random.uniform(-0.5, 0.5) * side_length
        else:
            u = np.random.uniform(-0.5, 0.5) * side_length
            v = np.random.choice([-0.5, 0.5]) * side_length
        p = u * tangent1 + v * tangent2 - offset * normal
        if noise > 0:
            p += np.random.normal(0, noise, size=3)
        points.append(p)

    points = np.array(points)
    np.savetxt(xyz_path, points, fmt="%.6f")
    param = np.concatenate([normal, [offset]])
    np.save(param_path, param)
    print(f"✅ 저장 완료: {xyz_path}, {param_path}")

# ✅ 2. Dataset 정의
class PlaneFittingDataset(Dataset):
    def __init__(self, xyz_file, param_file, normalize=True):
        self.points = np.loadtxt(xyz_file)
        self.gt_param = np.load(param_file)
        if normalize:
            self.points = self._normalize(self.points)
        self.points = torch.tensor(self.points, dtype=torch.float32)
        self.gt_param = torch.tensor(self.gt_param, dtype=torch.float32)

    def _normalize(self, pts):
        pts = pts - np.mean(pts, axis=0)
        scale = np.max(np.linalg.norm(pts, axis=1))
        return pts / scale

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.points, self.gt_param


# ✅ 3. MLP 모델 정의
class MLPPlanePredictor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 1000, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # [B, 1000, 3] → [B, 3000]
        return self.mlp(x)


# ✅ 4. point-to-plane loss 계산 함수
def compute_point_to_plane_loss(points, params):
    normal = params[:, :3]
    normal = torch.nn.functional.normalize(normal, dim=1)
    d = params[:, 3].unsqueeze(1)
    dot = (points * normal.unsqueeze(1)).sum(dim=-1)
    distance = torch.abs(dot + d)
    return distance.mean()


# ✅ 5. 학습 루프
def train():
    # ▶ 1단계: 데이터 생성
    generate_square_plane_pointcloud(
        xyz_path="LearnedPrior/square_plane_edge_heavy.xyz",
        param_path="LearnedPrior/square_plane_edge_heavy_param.npy",
        normal=[0, 0, 1],
        offset=0,
        num_points=1000,
        side_length=1.0,
        noise=0.0,
        edge_ratio=0.3
    )

    # ▶ 2단계: 데이터 로딩
    dataset = PlaneFittingDataset("square_plane.xyz", "square_plane_param.npy")
    loader = DataLoader(dataset, batch_size=1)

    # ▶ 3단계: 모델 & 옵티마이저 설정
    model = MLPPlanePredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    geo_weight = 1.0

    # ▶ 4단계: 학습 시작
    model.train()
    for epoch in range(1000):
        for points, gt_params in loader:
            pred_params = model(points)

            # 정규화
            pred_norm = pred_params[:, :3]
            pred_norm = pred_norm / pred_norm.norm(dim=1, keepdim=True)
            pred_params = torch.cat([pred_norm, pred_params[:, 3:]], dim=1)

            loss_param = nn.functional.mse_loss(pred_params, gt_params)
            loss_geo = compute_point_to_plane_loss(points, pred_params)
            loss = loss_param + geo_weight * loss_geo

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0 or epoch == 999:
            print(f"[Epoch {epoch}] Param Loss: {loss_param.item():.6f}, Geo Loss: {loss_geo.item():.6f}")

    # ▶ 5단계: 모델 저장
    torch.save(model.state_dict(), "mlp_plane_predictor.pth")
    print("✅ 모델 저장 완료: mlp_plane_predictor.pth")


# ✅ 6. 실행
if __name__ == "__main__":
    train()
