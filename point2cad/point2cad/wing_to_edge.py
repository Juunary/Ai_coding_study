#!/usr/bin/env python3
import numpy as np
import argparse
from scipy.spatial import cKDTree, ConvexHull
import plotly.graph_objects as go

def extract_edge_points_knn(points, k=10, percentile=95.0):
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k+1)
    kth_dist = distances[:, -1]
    thresh = np.percentile(kth_dist, percentile)
    edge_idx = np.where(kth_dist >= thresh)[0]
    return edge_idx, kth_dist, thresh


def find_corners(points, edge_idx):
    """
    PCA 투영 후 Convex Hull 순서로 정렬된 edge points에서
    각 꼭짓점의 내각을 계산해 90도에 가장 가까운 4점을 선택
    :param points: (N,3) numpy array 원본 점군
    :param edge_idx: 엣지 점들의 인덱스 리스트
    :return: corner_idx (원본 점 인덱스 리스트), hull sequence
    """
    edge_pts = points[edge_idx]
    # PCA 2D 투영
    centroid = edge_pts.mean(axis=0)
    centered = edge_pts - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pts2d = centered.dot(Vt[:2].T)
    hull = ConvexHull(pts2d)
    seq = hull.vertices.tolist()
    num = len(seq)
    deviations = []
    # 각 hull vertex에서 내각 계산
    for i, idx in enumerate(seq):
        prev_pt = edge_pts[seq[i-1]]
        curr_pt = edge_pts[idx]
        next_pt = edge_pts[seq[(i+1)%num]]
        v1 = prev_pt - curr_pt
        v2 = next_pt - curr_pt
        v1n = v1 / np.linalg.norm(v1)
        v2n = v2 / np.linalg.norm(v2)
        angle = np.degrees(np.arccos(np.clip(np.dot(v1n, v2n), -1.0, 1.0)))
        deviations.append((abs(angle - 90), idx))
    # 90도에 가장 가까운 4개 선택
    deviations.sort(key=lambda x: x[0])
    selected = [idx for _, idx in deviations[:8]]
    corner_idx = edge_idx[selected]
    return corner_idx, seq


def visualize(points, edge_idx, corner_idx, seq):
    colors = np.array(['lightgray'] * len(points), dtype=object)
    colors[edge_idx] = 'blue'
    fig = go.Figure()
    # 전체 점
    fig.add_trace(go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2],
                               mode='markers', marker=dict(size=3, color=colors), name='Points'))
    # edge outline
    loop = seq + [seq[0]]
    outline = points[edge_idx[loop]]
    fig.add_trace(go.Scatter3d(x=outline[:,0], y=outline[:,1], z=outline[:,2],
                               mode='lines', line=dict(color='rgba(0,0,255,0.5)', width=4),
                               name='Edge Outline'))
    # 4 corners
    fig.add_trace(go.Scatter3d(x=points[corner_idx,0], y=points[corner_idx,1], z=points[corner_idx,2],
                               mode='markers', marker=dict(size=8, color='red'), name='Corners'))
    fig.update_layout(
        title='Edge Outline with 4 Right-Angle Corners',
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    fig.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Highlight 4 right-angle corners on edge')
    parser.add_argument('--path_in', type=str, required=True, help='Input .xyz file')
    parser.add_argument('--k', type=int, default=10, help='k-th nearest neighbor')
    parser.add_argument('--percentile', type=float, default=95.0, help='Percentile threshold')
    args = parser.parse_args()

    pts = np.loadtxt(args.path_in)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError('Input must be N x >=3 array')
    points = pts[:, :3]
    edge_idx, kth_dist, thresh = extract_edge_points_knn(points, k=args.k, percentile=args.percentile)
    print(f'Edge points: {len(edge_idx)}/{len(points)}, threshold={thresh:.4f}')
    corner_idx, seq = find_corners(points, edge_idx)
    print(f'Selected 4 corners: {corner_idx.tolist()}')
    visualize(points, edge_idx, corner_idx, seq)
