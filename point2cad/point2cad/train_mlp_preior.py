# train_mlp_prior.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# ✅ 경로 설정
cube_xyz_path = "./assets/xyz/cube.xyz"
monalisa_param_path = "./assets/npy/plane_param_monalisa.npy"
save_path = "./mlp_prior_model/mlp_plane_predictor.pth"
os.makedirs("./mlp_prior_model", exist_ok=True)

# ✅ MLP 모델 정의
class MLPPlanePredictor(nn.Module):
    def __init__(self, input_dim, output_dim=6):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# ✅ 데이터 로딩
points = np.loadtxt(cube_xyz_path).astype(np.float32)  # (N, 3)
target = np.load(monalisa_param_path).astype(np.float32)  # (6,)

input_dim = points.shape[0] * 3
x = torch.from_numpy(points.reshape(1, -1))  # (1, N*3)
y = torch.from_numpy(target).reshape(1, -1)  # (1, 6)

# ✅ 모델, 손실 함수, 옵티마이저
model = MLPPlanePredictor(input_dim=input_dim)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ✅ 학습
for epoch in range(1000):
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"[{epoch}] Loss: {loss.item():.6f}")

# ✅ 저장
torch.save(model.state_dict(), save_path)
print(f"✅ MLP 모델 저장 완료: {save_path}")
