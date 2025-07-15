import numpy as np
import argparse
import math
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
##
##python .\wing_to_edge4.py --path_in ..\assets\xyz\panwing_gpu.xyz 4 --axis_offset 50 --axis_rotate_deg 90 --tilt_deg 40 
##
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

    # 90°에 가까운 내각 4개 선택
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

def rotation_matrix(axis, theta):
    axis = axis/np.linalg.norm(axis)
    K = np.array([[0,-axis[2],axis[1]],
                  [axis[2],0,-axis[0]],
                  [-axis[1],axis[0],0]])
    return np.eye(3)*np.cos(theta) + np.sin(theta)*K + (1-np.cos(theta))*np.outer(axis,axis)

def visualize_double(points,
                     e1, c1, s1, cent1, ax1,
                     e2, c2, s2, cent2, ax2,
                     copies, rot_axis, rot_center,
                     tilt_deg):
    fig = go.Figure()
    # 전체 점
    fig.add_trace(go.Scatter3d(
        x=points[:,0], y=points[:,1], z=points[:,2],
        mode='markers', marker=dict(size=2, color='lightgray'),
        showlegend=False
    ))
    # 회전축
    L = np.linalg.norm(points.max(axis=0)-points.min(axis=0))
    p1, p2 = rot_center-rot_axis*L, rot_center+rot_axis*L
    fig.add_trace(go.Scatter3d(
        x=[p1[0],p2[0]], y=[p1[1],p2[1]], z=[p1[2],p2[2]],
        mode='lines', line=dict(color='purple', width=3),
        name='Rotation Axis'
    ))

    # 원래 판 투영 좌표
    raw1, raw2 = points[e1], points[e2]
    w1 = (raw1-cent1).dot(ax1.T); proj1 = cent1 + w1.dot(ax1)
    wc1 = (points[c1]-cent1).dot(ax1.T); projc1 = cent1 + wc1.dot(ax1)
    w2 = (raw2-cent2).dot(ax2.T); proj2 = cent2 + w2.dot(ax2)
    wc2 = (points[c2]-cent2).dot(ax2.T); projc2 = cent2 + wc2.dot(ax2)
    loop1 = list(range(len(e1))) + [0]
    loop2 = list(range(len(e2))) + [0]

    # ─── tilt 적용 ───
    # 판의 법선 벡터
    plate_normal = np.cross(ax1[0], ax1[1])
    plate_normal /= np.linalg.norm(plate_normal)
    # tilt 축은 판 법선과 회전축의 외적
    tilt_axis = np.cross(plate_normal, rot_axis)
    tilt_axis /= np.linalg.norm(tilt_axis)
    tilt_rad = math.radians(tilt_deg)
    T = rotation_matrix(tilt_axis, tilt_rad)
    # cent1/cent2 기준으로 proj 회전
    proj1  = cent1 + (proj1  - cent1)  @ T.T
    projc1 = cent1 + (projc1 - cent1)  @ T.T
    proj2  = cent2 + (proj2  - cent2)  @ T.T
    projc2 = cent2 + (projc2 - cent2)  @ T.T
    # ─────────────────

    for k in range(copies):
        θ = 2*math.pi * k / copies
        R = rotation_matrix(rot_axis, θ)

        # Front
        pts1 = ((proj1-rot_center)@R.T)+rot_center
        pts1 = pts1[loop1]
        rc1  = ((projc1-rot_center)@R.T)+rot_center
        fig.add_trace(go.Scatter3d(
            x=pts1[:,0], y=pts1[:,1], z=pts1[:,2],
            mode='lines', line=dict(color='blue', width=4),
            name='Front Outline', legendgroup='F', showlegend=(k==0)
        ))
        fig.add_trace(go.Scatter3d(
            x=rc1[:,0], y=rc1[:,1], z=rc1[:,2],
            mode='markers', marker=dict(size=6, color='red'),
            name='Front Corners', legendgroup='F', showlegend=(k==0)
        ))
        # Back
        pts2 = ((proj2-rot_center)@R.T)+rot_center
        pts2 = pts2[loop2]
        rc2  = ((projc2-rot_center)@R.T)+rot_center
        fig.add_trace(go.Scatter3d(
            x=pts2[:,0], y=pts2[:,1], z=pts2[:,2],
            mode='lines', line=dict(color='green', width=4),
            name='Back Outline', legendgroup='B', showlegend=(k==0)
        ))
        fig.add_trace(go.Scatter3d(
            x=rc2[:,0], y=rc2[:,1], z=rc2[:,2],
            mode='markers', marker=dict(size=6, color='orange'),
            name='Back Corners', legendgroup='B', showlegend=(k==0)
        ))

    fig.update_layout(
        title=f'Rotated Copies: {copies}×',
        scene=dict(aspectmode='data',
                   xaxis=dict(visible=False),
                   yaxis=dict(visible=False),
                   zaxis=dict(visible=False)),
        margin=dict(l=0,r=0,b=0,t=40)
    )
    fig.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in',         type=str, required=True, help='Input .xyz')
    parser.add_argument('copies',            type=int, nargs='?', default=1,  help='복제 개수')
    parser.add_argument('--axis_offset',     type=float, default=None,    help='오프셋 거리')
    parser.add_argument('--axis_rotate_deg', type=float, default=0.0,     help='축 회전 각도(°)')
    parser.add_argument('--tilt_deg',        type=float, default=0.0,     help='판 기울기 각도(°)')
    args = parser.parse_args()

    pts = np.loadtxt(args.path_in)
    if pts.ndim!=2 or pts.shape[1]<3:
        raise ValueError('Input must be N x >=3 array')
    pts = pts[:,:3]

    # …(중략: separate_faces, extract_face_outline_and_corners)…
    idx_f, idx_b, axis, center, thick = separate_faces(pts)
    e1, c1, s1, cent1, ax1 = extract_face_outline_and_corners(pts, idx_f)
    if args.axis_rotate_deg:
        Rr = rotation_matrix(ax1[1], math.radians(args.axis_rotate_deg))
        axis = Rr.dot(axis)
    offset = thick/2 if args.axis_offset is None else args.axis_offset
    rot_center = center + ax1[1] * offset
    e2, c2, s2, cent2, ax2 = extract_face_outline_and_corners(pts, idx_b)

    visualize_double(pts,
                     e1, c1, s1, cent1, ax1,
                     e2, c2, s2, cent2, ax2,
                     args.copies, axis, rot_center,
                     args.tilt_deg)