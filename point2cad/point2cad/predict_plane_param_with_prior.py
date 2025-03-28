import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
from point2cad.utils import normalize_points

# ✅ 모델 정의 (cube_dense 학습 시와 동일하게!)
class MLPPlanePredictor(torch.nn.Module):
    def __init__(self, input_dim=3 * 5400, output_dim=6):  # 🔸 5400 = cube_dense 점 수
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# 🔹 경로 설정
input_path = "./assets/xyz/abc_monalisa.xyz"
model_path = "./mlp_prior_model/mlp_plane_predictor.pth"
save_path = "./assets/npy/plane_param_monalisa_pred.npy"

# 🔸 1. monalisa 점군 불러오기 및 정규화
points = np.loadtxt(input_path).astype(np.float32)
points = normalize_points(points)

# 🔸 2. cube와 동일한 개수로 자르기 (cube_dense는 6면 × 30×30 = 5400개 점)
points = points[:5400]  # 포인트 수가 부족하다면 여기서 에러 나지 않게 조심

# 🔸 3. 모델 로드
device = torch.device("cpu")
model = MLPPlanePredictor()
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 🔸 4. 예측
input_tensor = torch.from_numpy(points.reshape(1, -1)).float().to(device)
with torch.no_grad():
    pred_param = model(input_tensor).cpu().numpy().squeeze()  # shape: (6,)

# 🔸 5. 저장
np.save(save_path, pred_param)
print(f"✅ 예측된 plane parameter 저장 완료: {save_path}")
print("🔹 Normal Vector:", pred_param[:3])
print("🔹 Center Point:", pred_param[3:])
