import numpy as np
import os

def generate_cylinder_pointcloud_with_caps(xyz_path, param_path,
                                            axis=[0, 0, 1],
                                            center=[0, 0, 0],
                                            radius=1.0,
                                            height=2.0,
                                            num_points=1000,
                                            cap_ratio=0.2,
                                            noise=0.0):
    """
    - cap_ratio: 전체 점 중 윗/아랫면 뚜껑에 배치할 비율 (0.2 → 20%)
    """
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    center = np.array(center)

    # 직교 벡터 구하기
    if np.allclose(axis, [0, 0, 1]):
        tangent1 = np.array([1, 0, 0])
    else:
        tangent1 = np.cross(axis, [0, 0, 1])
        tangent1 = tangent1 / np.linalg.norm(tangent1)
    tangent2 = np.cross(axis, tangent1)

    # 점 개수 배분
    num_caps = int(num_points * cap_ratio)  # 총 뚜껑 점 개수
    num_cap_each = num_caps // 2
    num_side = num_points - num_caps

    points = []

    # ▶ 원기둥 곡면
    for _ in range(num_side):
        theta = np.random.uniform(0, 2 * np.pi)
        h = np.random.uniform(-0.5, 0.5) * height
        circle_pt = radius * (np.cos(theta) * tangent1 + np.sin(theta) * tangent2)
        axial_offset = h * axis
        p = center + circle_pt + axial_offset
        if noise > 0:
            p += np.random.normal(0, noise, size=3)
        points.append(p)

    # ▶ 아랫면 원판 (bottom cap)
    for _ in range(num_cap_each):
        r = np.sqrt(np.random.uniform(0, 1)) * radius  # 균일하게 원판 채우기
        theta = np.random.uniform(0, 2 * np.pi)
        circle_pt = r * (np.cos(theta) * tangent1 + np.sin(theta) * tangent2)
        axial_offset = -0.5 * height * axis
        p = center + circle_pt + axial_offset
        if noise > 0:
            p += np.random.normal(0, noise, size=3)
        points.append(p)

    # ▶ 윗면 원판 (top cap)
    for _ in range(num_cap_each):
        r = np.sqrt(np.random.uniform(0, 1)) * radius
        theta = np.random.uniform(0, 2 * np.pi)
        circle_pt = r * (np.cos(theta) * tangent1 + np.sin(theta) * tangent2)
        axial_offset = 0.5 * height * axis
        p = center + circle_pt + axial_offset
        if noise > 0:
            p += np.random.normal(0, noise, size=3)
        points.append(p)

    points = np.array(points)

    # 폴더 없으면 생성
    os.makedirs(os.path.dirname(xyz_path), exist_ok=True)

    np.savetxt(xyz_path, points, fmt="%.6f")
    param = np.concatenate([axis, center, [radius]])
    np.save(param_path, param)

    print(f"✅ 저장 완료: {xyz_path}, {param_path} (총 {len(points)}개 점)")


# ✅ 실행 예시
if __name__ == "__main__":
    generate_cylinder_pointcloud_with_caps(
        xyz_path="../cylinder_with_caps.xyz",
        param_path="../cylinder_with_caps_param.npy",
        axis=[0, 0, 1],
        center=[0, 0, 0],
        radius=0.5,
        height=2.0,
        num_points=10000,
        cap_ratio=0.2,  # 20%는 윗면·아랫면
        noise=0.0
    )
