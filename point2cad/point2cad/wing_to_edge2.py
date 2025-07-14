#!/usr/bin/env python3
import numpy as np
import argparse
from scipy.spatial import ConvexHull
import plotly.graph_objects as go

def separate_faces(points, n_iters=10):
    """
    points: (N,3)
    returns: idx_front, idx_back
    """
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = Vt[-1]  # 두께 방향 법선
    depth = (points - centroid).dot(normal)

    # 1D K-Means(2) on depth
    c1, c2 = depth.min(), depth.max()
    for _ in range(n_iters):
        labels = np.where(np.abs(depth - c1) < np.abs(depth - c2), 0, 1)
        new_c1 = depth[labels == 0].mean() if np.any(labels == 0) else c1
        new_c2 = depth[labels == 1].mean() if np.any(labels == 1) else c2
        if np.isclose(new_c1, c1) and np.isclose(new_c2, c2):
            break
        c1, c2 = new_c1, new_c2

    idx_front = np.where(labels == 0)[0]
    idx_back  = np.where(labels == 1)[0]
    return idx_front, idx_back

def extract_face_outline_and_corners(points, idx_cluster):
    """
    points: (N,3)
    idx_cluster: 원본 인덱스 리스트
    returns:
      edge_idx: 원본 인덱스 리스트 (hull outline)
      corner_idx: np.array (4개의 모서리 점 인덱스)
      seq: cluster-relative hull 순서
      centroid_sub: 해당 면 점군의 중심
      axes2: 해당 면 평면의 두 PCA 축 (shape=(2,3))
    """
    sub_pts = points[idx_cluster]
    # 1) 해당 면 전용 PCA 축 계산
    centroid_sub = sub_pts.mean(axis=0)
    _, _, Vt_sub = np.linalg.svd(sub_pts - centroid_sub, full_matrices=False)
    axes2 = Vt_sub[:2]                   # 평면 축 2개 (3D 벡터)
    pts2d = (sub_pts - centroid_sub).dot(axes2.T)

    # 2) Convex Hull
    hull = ConvexHull(pts2d)
    seq = hull.vertices.tolist()         # cluster-relative 인덱스

    # 3) 각 hull vertex에서 내각 계산
    hull2d = pts2d[seq]
    n = len(seq)
    deviations = []
    for i in range(n):
        prev_pt = hull2d[(i-1) % n]
        curr_pt = hull2d[i]
        next_pt = hull2d[(i+1) % n]
        v1 = prev_pt - curr_pt
        v2 = next_pt - curr_pt
        v1n = v1 / np.linalg.norm(v1)
        v2n = v2 / np.linalg.norm(v2)
        angle = np.degrees(np.arccos(np.clip(np.dot(v1n, v2n), -1.0, 1.0)))
        deviations.append((abs(angle - 90.0), i))

    # 4) 내각이 90°에 가장 가까운 4개 선택
    deviations.sort(key=lambda x: x[0])
    corner_idxs_rel = [idx for _, idx in deviations[:4]]
    corner_idx = np.array([ idx_cluster[seq[i]] for i in corner_idxs_rel ])

    edge_idx = idx_cluster[seq]
    return edge_idx, corner_idx, seq, centroid_sub, axes2

def visualize_double(points,
                     edge1, corner1, seq1, cent1, axes1,
                     edge2, corner2, seq2, cent2, axes2):
    fig = go.Figure()

    # 전체 점 (회색)
    fig.add_trace(go.Scatter3d(
        x=points[:,0], y=points[:,1], z=points[:,2],
        mode='markers',
        marker=dict(size=2, color='lightgray'),
        name='All Points'
    ))

    # Front outline (파랑)
    raw1 = points[edge1]
    w1 = (raw1 - cent1).dot(axes1.T)       # (M,2)
    proj1 = cent1 + w1.dot(axes1)          # (M,3)
    loop1 = list(range(len(edge1))) + [0]
    pts1 = proj1[loop1]
    fig.add_trace(go.Scatter3d(
        x=pts1[:,0], y=pts1[:,1], z=pts1[:,2],
        mode='lines',
        line=dict(color='blue', width=4),
        name='Front Outline'
    ))

    # Back outline (초록)
    raw2 = points[edge2]
    w2 = (raw2 - cent2).dot(axes2.T)
    proj2 = cent2 + w2.dot(axes2)
    loop2 = list(range(len(edge2))) + [0]
    pts2 = proj2[loop2]
    fig.add_trace(go.Scatter3d(
        x=pts2[:,0], y=pts2[:,1], z=pts2[:,2],
        mode='lines',
        line=dict(color='green', width=4),
        name='Back Outline'
    ))

    # Front corners (빨강)
    raw_c1 = points[corner1]
    w_c1 = (raw_c1 - cent1).dot(axes1.T)
    proj_c1 = cent1 + w_c1.dot(axes1)
    fig.add_trace(go.Scatter3d(
        x=proj_c1[:,0], y=proj_c1[:,1], z=proj_c1[:,2],
        mode='markers',
        marker=dict(size=6, color='red'),
        name='Front Corners'
    ))

    # Back corners (주황)
    raw_c2 = points[corner2]
    w_c2 = (raw_c2 - cent2).dot(axes2.T)
    proj_c2 = cent2 + w_c2.dot(axes2)
    fig.add_trace(go.Scatter3d(
        x=proj_c2[:,0], y=proj_c2[:,1], z=proj_c2[:,2],
        mode='markers',
        marker=dict(size=6, color='orange'),
        name='Back Corners'
    ))

    fig.update_layout(
        title='Thin Box: 2 Faces Edge & Corners',
        scene=dict(xaxis=dict(visible=False),
                   yaxis=dict(visible=False),
                   zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', type=str, required=True, help='Input .xyz file')
    args = parser.parse_args()

    pts = np.loadtxt(args.path_in)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError('Input must be N x >=3 array')
    points = pts[:, :3]

    # 1) 앞·뒤 면 분리
    idx_f, idx_b = separate_faces(points)

    # 2) 각 면에서 outline & corners(내각 기반) 추출
    e1, c1, seq1, cent1, ax1 = extract_face_outline_and_corners(points, idx_f)
    e2, c2, seq2, cent2, ax2 = extract_face_outline_and_corners(points, idx_b)

    print(f'Front face: outline pts={len(e1)}, corners={c1.tolist()}')
    print(f'Back  face: outline pts={len(e2)}, corners={c2.tolist()}')

    # 3) 시각화
    visualize_double(points,
                     e1, c1, seq1, cent1, ax1,
                     e2, c2, seq2, cent2, ax2)
