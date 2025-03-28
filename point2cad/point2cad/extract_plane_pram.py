import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np

from point2cad.primitive_forward import Fit
from point2cad.utils import normalize_points

# 입력: 단순 평면의 포인트 클라우드 파일 (.xyz)
input_path = "./assets/xyz/abc_monalisa.xyz"  
output_dir = "./learned_prior_params"
os.makedirs(output_dir, exist_ok=True)

# 1. 포인트 불러오기
points = np.loadtxt(input_path).astype(np.float32)

# 2. 정규화
points = normalize_points(points)

# 3. 텐서 변환
points_tensor = torch.from_numpy(points)
weights = torch.ones((points_tensor.shape[0], 1), dtype=torch.float32)
normals_dummy = torch.zeros_like(points_tensor)

# 4. 피팅 수행
fit = Fit()
normal_tensor, d_scalar = fit.fit_plane_torch(points_tensor, normals_dummy, weights)

# 5. 결과 추출
normal = normal_tensor[0].cpu().numpy()         # (3,)
d = d_scalar.item()
center = (normal * d).astype(np.float32)        # 중심점 계산

# 6. 저장
param_vector = np.concatenate([normal, center])  # [nx, ny, nz, cx, cy, cz]
np.save(os.path.join(output_dir, "plane_param_monalisa.npy"), param_vector)

# 7. 출력
print("✅ plane_param.npy 저장 완료:")
print("  Normal Vector:", normal)
print("  Center Point:", center)
