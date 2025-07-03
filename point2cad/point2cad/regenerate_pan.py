import numpy as np
import sys
import os

def find_most_perpendicular_axis(data):
    # 레이어 0,1 점만 추출
    layer_mask = np.logical_or(data[:,3]==0, data[:,3]==1)
    pts = data[layer_mask, :3]
    pts_centered = pts - np.mean(pts, axis=0)
    # PCA로 주법선(분산 가장 작은 축) 추출
    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal_vec = eigvecs[:, np.argmin(eigvals)]  # 평면 법선
    # XYZ축 unit vector와의 내적(절댓값) → 가장 0에 가까운 축이 가장 수직
    axes = np.eye(3)  # [[1,0,0],[0,1,0],[0,0,1]]
    dots = np.abs(axes @ normal_vec)
    axis = np.argmin(dots)  # 가장 수직인 축
    return axis

def process_pan_auto_axis(xyzc_path, thickness=0.2, float_fmt="%.8f"):
    data = np.loadtxt(xyzc_path)
    layer_col = data[:, 3]
    unique_layers = np.unique(layer_col)
    two_largest = unique_layers[-2:]

    # 레이어 리맵: 가장 큰 값 → 0, 두 번째 → 1, 나머지 → 2
    mask0 = layer_col == two_largest[1]
    mask1 = layer_col == two_largest[0]
    new_layer = np.full_like(layer_col, 2)
    new_layer[mask0] = 0
    new_layer[mask1] = 1
    data[:, 3] = new_layer

    # 압축축 자동 감지 (법선 벡터와 가장 수직인 축)
    axis = find_most_perpendicular_axis(data)
    axis_name = ['X', 'Y', 'Z'][axis]
    print(f"[i] 평면 법선과 가장 수직인 축({axis_name}축, axis={axis})을 압축 방향으로 선택")

    # 2번(수직면) 레이어만 해당 축으로 두께 압축
    mask2 = data[:, 3] == 2
    if np.sum(mask2) > 0:
        axis_vals = data[mask2, axis]
        center = np.median(axis_vals)
        min_val = center - thickness/2
        max_val = center + thickness/2
        data[mask2, axis] = np.clip(axis_vals, min_val, max_val)

    # 파일명 '_thin' 붙여 저장
    base, ext = os.path.splitext(xyzc_path)
    out_path = base + "_thin" + ext

    np.savetxt(out_path, data, fmt=f"{float_fmt} {float_fmt} {float_fmt} %d")
    print(f"[✔] 수직면(엣지)만 {axis_name}축으로 얇게 압축하여 {out_path}에 저장 (두께 {thickness})")
    return out_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python regenerate_pan.py <입력.xyzc>")
        exit(1)
    in_file = sys.argv[1]
    process_pan_auto_axis(in_file, thickness=0.2, float_fmt="%.8f")
