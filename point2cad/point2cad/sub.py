import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import trimesh

from point2cad.primitive_forward import Fit
from point2cad.utils import normalize_points

# ✅ 1. 경로 설정
param_path = "./assets/npy/plane_param_monalisa.npy"  # 예측된 파라미터
output_ply_path = "./out/before_based_mesh.ply"

# ✅ 2. 파라미터 로드
param = np.load(param_path)  # [nx, ny, nz, cx, cy, cz]
normal = param[:3]
center = param[3:]

# ✅ 3. 평면 샘플링 (Point2CAD의 내부 샘플링 함수 활용)
fit = Fit()
plane_points = fit.sample_plane(d=np.dot(normal, center), n=normal, mean=center)

# ✅ 4. 삼각형 생성 (grid 해석: 120 x 120이므로 정사각형 119x119개)
res = 120
faces = []
for i in range(res - 1):
    for j in range(res - 1):
        idx = i * res + j
        faces.append([idx, idx + 1, idx + res])
        faces.append([idx + 1, idx + res + 1, idx + res])
faces = np.array(faces)

# ✅ 5. 메쉬 저장
mesh = trimesh.Trimesh(vertices=plane_points, faces=faces)
os.makedirs(os.path.dirname(output_ply_path), exist_ok=True)
mesh.export(output_ply_path)
print(f"✅ 저장 완료: {output_ply_path}")
