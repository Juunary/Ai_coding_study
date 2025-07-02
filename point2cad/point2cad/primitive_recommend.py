import numpy as np
import argparse
import open3d as o3d
from sklearn.decomposition import PCA

from primitive_utils import fitcylinder

# 1. 점군 로드
def load_xyz(filename):
    pts = []
    with open(filename, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            vals = list(map(float, line.strip().split()))
            pts.append(vals[:3])
    return np.array(pts)

# 2. Cylinder fitting 적용
points = load_xyz('your_segment.xyz')

# fit_cylinder 반환 형식에 따라 아래 코드 조정 필요
axis, center, radius, residual = fitcylinder(points)

print(f'Cylinder axis (축): {axis}')
print(f'Cylinder center (중심): {center}')
print(f'Cylinder radius (반지름): {radius}')
print(f'Fitting residual (평균 오차): {residual}')


def load_xyz(filename):
    pts = []
    with open(filename, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            vals = list(map(float, line.strip().split()))
            pts.append(vals[:3])
    return np.array(pts)

def recommend_primitive(points, knn=30):
    # PCA로 주법선 구하기
    points_centered = points - points.mean(axis=0)
    pca = PCA(n_components=3)
    pca.fit(points_centered)
    normal = pca.components_[-1]

    # open3d로 법선 추정
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    normals = np.asarray(pcd.normals)

    # 법선 분산 분석
    normals_centered = normals - normals.mean(axis=0)
    cov = np.cov(normals_centered.T)
    eigvals, _ = np.linalg.eigh(cov)

    # 결과 해석
    type_msg = ""
    if eigvals[2] < 0.02:
        type_msg = "평면(Plane) fitting 추천"
    elif eigvals[1] < 0.02:
        type_msg = "원통(Cylinder) fitting 추천"
    else:
        type_msg = "이차곡면(Paraboloid/구 등) fitting 추천"

    print(f"주법선 벡터: {normal}")
    print(f"법선 분산 고유값: {eigvals}")
    print(f"추천 프리미티브 타입: {type_msg}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', type=str, required=True, help='입력 xyz 파일 경로')
    args = parser.parse_args()

    pts = load_xyz(args.path_in)
    print(f"포인트 개수: {len(pts)}")
    recommend_primitive(pts)

if __name__ == "__main__":
    main()
