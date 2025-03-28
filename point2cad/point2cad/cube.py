import numpy as np

def sample_square_face(p1, p2, p3, p4, resolution=20):
    """
    네 꼭짓점(p1, p2, p3, p4)으로 정의된 사각형 면을 resolution x resolution 만큼 샘플링
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    p4 = np.array(p4)

    points = []
    for i in range(resolution):
        for j in range(resolution):
            s = i / (resolution - 1)
            t = j / (resolution - 1)
            point = (1 - s) * (1 - t) * p1 + s * (1 - t) * p2 + s * t * p3 + (1 - s) * t * p4
            points.append(point)
    return np.array(points)

# 8개 꼭짓점
vertices = np.array([
    [10, -10, -10],
    [-10, -10, -10],
    [-10, 10, -10],
    [10, 10, -10],
    [-10, 10, 10],
    [10, 10, 10],
    [-10, -10, 10],
    [10, -10, 10]
])

# 6개 면 정의 (시계 방향)
faces = [
    [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
    [vertices[4], vertices[5], vertices[7], vertices[6]],  # top
    [vertices[1], vertices[6], vertices[4], vertices[2]],  # left
    [vertices[0], vertices[3], vertices[5], vertices[7]],  # right
    [vertices[2], vertices[4], vertices[5], vertices[3]],  # front
    [vertices[0], vertices[7], vertices[6], vertices[1]],  # back
]

all_points = []
for face in faces:
    sampled = sample_square_face(*face, resolution=30)
    all_points.append(sampled)

point_cloud = np.vstack(all_points)

# 저장
np.savetxt("cube_dense.xyz", point_cloud, fmt="%.6f")
