# 파일명: fit_and_mirror_caps.py
import argparse
import os
import numpy as np
import torch
from fitting_one_surface import fit_basic_primitives

def mirror_across_cylinder_surface(pts: np.ndarray,
                                   axis_point: np.ndarray,
                                   axis_dir: np.ndarray,
                                   radius: float) -> np.ndarray:
    """
    실린더 표면을 기준으로 pts를 대칭(반사)시킨 점군 반환
    """
    # 축 방향 단위 벡터
    a = axis_dir / np.linalg.norm(axis_dir)
    # 원점 이동
    v = pts - axis_point
    # 축 성분 및 수직 성분 분리
    proj_len = np.dot(v, a)
    v_para = np.outer(proj_len, a)
    v_perp = v - v_para
    # 수직 성분 거리
    d = np.linalg.norm(v_perp, axis=1)
    # 수직 방향 단위 벡터
    with np.errstate(invalid='ignore', divide='ignore'):
        r_hat = v_perp / d[:, None]
        r_hat[np.isnan(r_hat)] = 0
    # 반사된 거리
    d_ref = 2 * radius - d
    # 반사 포인트 계산
    v_perp_ref = r_hat * d_ref[:, None]
    pts_ref = axis_point + v_para + v_perp_ref
    return pts_ref


def sample_caps(center: np.ndarray, axis: np.ndarray, radius: float, t_extremes: list, num_samples: int = 100) -> np.ndarray:
    """
    t_extremes: 축 상 위치 [t_min, t_max]
    각 위치에서 반지름 r인 원형 포인트 num_samples 생성
    """
    a = axis / np.linalg.norm(axis)
    # 축에 수직인 임의 벡터
    tmp = np.array([1.0, 0.0, 0.0])
    if np.allclose(a, tmp):
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(a, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    circles = []
    thetas = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
    for t in t_extremes:
        center_plane = center + a * t
        circle = center_plane[None, :] + radius * (
            np.outer(np.cos(thetas), u) + np.outer(np.sin(thetas), v)
        )
        circles.append(circle)
    return np.vstack(circles)


def main():
    parser = argparse.ArgumentParser(
        description="곡면 입력 → 방정식 출력 → 반사곡면 생성(cylinder surface) → 캡 추가 → 결과 .xyz 저장"
    )
    parser.add_argument('--path_in', required=True, help='입력 .xyz 경로')
    parser.add_argument('--scale', type=float, default=1.0, help='실린더 반지름 배율 (기본 1)')
    parser.add_argument('--caps', action='store_true', help='끝단에 원형 캡 추가')
    args = parser.parse_args()

    # 1) xyz 로드
    data = np.loadtxt(args.path_in)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"'{args.path_in}' 파일이 N×3 형태가 아닙니다.")
    pts = data[:, :3]

    # 2) primitive 피팅 (cylinder)
    pts_tensor = torch.from_numpy(pts).float()
    shapes = fit_basic_primitives(pts_tensor)
    axis, center, radius = shapes['cylinder_params']
    axis = np.array(axis)
    center = np.array(center)

    # 3) radius 확대
    radius *= args.scale

    # 4) 실린더 방정식 출력
    a_norm = axis / np.linalg.norm(axis)
    cx, cy, cz = center
    ax, ay, az = a_norm
    print("\n=== Scaled Cylinder Equation ===")
    print(
        f"||(p - [{cx:.6f},{cy:.6f},{cz:.6f}]) - (((p - [{cx:.6f},{cy:.6f},{cz:.6f}])·[{ax:.6f},{ay:.6f},{az:.6f}]) * [{ax:.6f},{ay:.6f},{az:.6f}])|| = {radius:.6f}"
    )

    # 5) 반사 곡면 생성 (cylinder surface)
    pts_reflect = mirror_across_cylinder_surface(pts, center, axis, radius)

    # 6) 캡 생성 옵션
    result = [pts, pts_reflect]
    if args.caps:
        v = pts - center
        t_vals = np.dot(v, a_norm)
        t_min, t_max = t_vals.min(), t_vals.max()
        caps = sample_caps(center, axis, radius, [t_min, t_max])
        result.append(caps)

    # 7) 합치기 및 저장
    all_pts = np.vstack(result)
    in_dir = os.path.dirname(args.path_in)
    name = os.path.splitext(os.path.basename(args.path_in))[0]
    suffix = '_CapReflect' if args.caps else '_Reflect'
    out_name = f"{name}{suffix}.xyz"
    path_out = os.path.join(in_dir or '.', out_name)
    np.savetxt(path_out, all_pts, fmt="%.6f")
    print(f"✔ 저장 완료: {path_out}")

if __name__ == '__main__':
    main()
