#!/usr/bin/env python3
import numpy as np
import argparse
import math
from scipy.spatial import ConvexHull
import plotly.graph_objects as go

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians using the Rodrigues' rotation formula.
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad),   2*(bd-ac)],
                     [2*(bc-ad),   aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac),   2*(cd-ab),   aa+dd-bb-cc]])

def separate_faces(points, n_iters=10):
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = Vt[-1]
    depth = (points - centroid).dot(normal)
    thickness = depth.max() - depth.min()

    c1, c2 = depth.min(), depth.max()
    for _ in range(n_iters):
        labels = np.where(np.abs(depth - c1) < np.abs(depth - c2), 0, 1)
        nc1 = depth[labels==0].mean() if np.any(labels==0) else c1
        nc2 = depth[labels==1].mean() if np.any(labels==1) else c2
        if np.isclose(nc1, c1) and np.isclose(nc2, c2): break
        c1, c2 = nc1, nc2

    idx_f = np.where(labels==0)[0]
    idx_b = np.where(labels==1)[0]
    return idx_f, idx_b, normal, centroid, thickness

def extract_face_outline_and_corners(points, idx_cluster):
    sub = points[idx_cluster]
    cent = sub.mean(axis=0)
    _, _, Vt = np.linalg.svd(sub - cent, full_matrices=False)
    axes2 = Vt[:2]
    pts2d = (sub - cent).dot(axes2.T)

    hull = ConvexHull(pts2d)
    seq = hull.vertices.tolist()
    # 내각 기반 코너 4개
    hull2d = pts2d[seq]
    dev = []
    n = len(seq)
    for i in range(n):
        p0, p1, p2 = hull2d[(i-1)%n], hull2d[i], hull2d[(i+1)%n]
        v1, v2 = p0-p1, p2-p1
        a1, a2 = v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)
        ang = np.degrees(np.arccos(np.clip(np.dot(a1,a2), -1,1)))
        dev.append((abs(ang-90), i))
    dev.sort(key=lambda x: x[0])
    corner_rel = [i for _,i in dev[:4]]
    corner_idx = np.array([ idx_cluster[seq[i]] for i in corner_rel ])

    edge_idx = idx_cluster[seq]
    return edge_idx, corner_idx, seq, cent, axes2

def visualize_cone(points,
                   e1, c1, seq1, cent1, ax1,
                   e2, c2, seq2, cent2, ax2,
                   copies, axis, cent_global,
                   base_radius, cone_height):
    fig = go.Figure()
    # 전체 점
    fig.add_trace(go.Scatter3d(
        x=points[:,0], y=points[:,1], z=points[:,2],
        mode='markers', marker=dict(size=2, color='lightgray'),
        showlegend=False
    ))
    # 회전축(가상의 apex→base line)
    apex = cent1
    base = cent1 + axis*cone_height
    fig.add_trace(go.Scatter3d(
        x=[apex[0],base[0]],
        y=[apex[1],base[1]],
        z=[apex[2],base[2]],
        mode='lines', line=dict(color='purple', width=3),
        name='Cone Axis'
    ))

    # 축에 수직 평면 basis
    e_r = ax1[0]                           # 앞면 PCA 첫 축
    e_t = np.cross(axis, e_r)
    e_t /= np.linalg.norm(e_t)

    # 원본 2D 투영 좌표 (앞/뒤)
    raw1   = points[e1]; w1   = (raw1-cent1).dot(ax1.T); proj1  = cent1 + w1.dot(ax1)
    raw2   = points[e2]; w2   = (raw2-cent2).dot(ax2.T); proj2  = cent2 + w2.dot(ax2)
    rawc1  = points[c1]; wc1  = (rawc1-cent1).dot(ax1.T); projc1 = cent1 + wc1.dot(ax1)
    rawc2  = points[c2]; wc2  = (rawc2-cent2).dot(ax2.T); projc2 = cent2 + wc2.dot(ax2)
    loop1 = list(range(len(e1)))+[0]
    loop2 = list(range(len(e2)))+[0]

    for k in range(copies):
        t = k/(copies-1) if copies>1 else 0.0
        zk = cone_height * t      # 축 방향 위치
        rk = base_radius * t      # 반지름
        θ  = 2*math.pi * k / copies
        # 각 clone 중심
        pos = apex + axis*zk \
              + (math.cos(θ)*e_r + math.sin(θ)*e_t) * rk
        # shift 벡터
        shift1 = pos - cent1
        shift2 = pos - cent2

        # Front plate
        pts1 = proj1 + shift1
        fig.add_trace(go.Scatter3d(
            x=pts1[loop1,0], y=pts1[loop1,1], z=pts1[loop1,2],
            mode='lines', line=dict(color='blue', width=4),
            name='Front Outline', legendgroup='F', showlegend=(k==0)
        ))
        rc1 = projc1 + shift1
        fig.add_trace(go.Scatter3d(
            x=rc1[:,0], y=rc1[:,1], z=rc1[:,2],
            mode='markers', marker=dict(size=6, color='red'),
            name='Front Corners', legendgroup='F', showlegend=(k==0)
        ))

        # Back plate
        pts2 = proj2 + shift2
        fig.add_trace(go.Scatter3d(
            x=pts2[loop2,0], y=pts2[loop2,1], z=pts2[loop2,2],
            mode='lines', line=dict(color='green', width=4),
            name='Back Outline', legendgroup='B', showlegend=(k==0)
        ))
        rc2 = projc2 + shift2
        fig.add_trace(go.Scatter3d(
            x=rc2[:,0], y=rc2[:,1], z=rc2[:,2],
            mode='markers', marker=dict(size=6, color='orange'),
            name='Back Corners', legendgroup='B', showlegend=(k==0)
        ))

    fig.update_layout(
        title=f'Conical Copies: {copies}×',
        scene=dict(aspectmode='data',
                   xaxis=dict(visible=False),
                   yaxis=dict(visible=False),
                   zaxis=dict(visible=False)),
        margin=dict(l=0,r=0,b=0,t=40)
    )
    fig.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in',        type=str, required=True,
                        help='Input .xyz 파일')
    parser.add_argument('copies',           type=int, nargs='?', default=1,
                        help='복제 개수')
    parser.add_argument('--base_radius',    type=float, required=True,
                        help='콘 밑면 반지름 R')
    parser.add_argument('--cone_height',    type=float, default=None,
                        help='콘 높이 H (미지정 시 두 면 간 거리 사용)')
    parser.add_argument('--axis_rotate_deg',type=float, default=0.0,
                        help='법선축 회전 각도(°)')  # 기존 옵션 그대로 유지
    args = parser.parse_args()

    pts = np.loadtxt(args.path_in)
    if pts.ndim!=2 or pts.shape[1]<3:
        raise ValueError('Input must be N x >=3 array')
    pts = pts[:,:3]

    # 1) 앞/뒤 면 분리
    idx_f, idx_b, axis, center, thick = separate_faces(pts)
    # 2) 앞면 PCA 축 미리 구하기
    e1, c1, s1, cent1, ax1 = extract_face_outline_and_corners(pts, idx_f)
    # 3) 축 회전 옵션
    if args.axis_rotate_deg:
        Rrot = rotation_matrix(ax1[1], np.deg2rad(args.axis_rotate_deg))
        axis = Rrot.dot(axis)
    # 4) 콘 높이 H 설정
    H = args.cone_height if args.cone_height is not None else np.dot((extract_face_outline_and_corners(pts, idx_b)[3] - cent1), axis)
    # 5) 뒷면 정보
    e2, c2, s2, cent2, ax2 = extract_face_outline_and_corners(pts, idx_b)

    print(f'Front: outline pts={len(e1)}, corners={c1.tolist()}')
    print(f'Back : outline pts={len(e2)}, corners={c2.tolist()}')

    # 6) 원뿔형 복제 시각화
    visualize_cone(pts,
                   e1, c1, s1, cent1, ax1,
                   e2, c2, s2, cent2, ax2,
                   args.copies, axis, cent1,
                   args.base_radius, H)
