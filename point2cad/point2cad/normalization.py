#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
import open3d as o3d

# ---------- 선형대수 유틸 ----------

def _skew(v):
    x, y, z = v
    return np.array([[0, -z,  y],
                     [z,  0, -x],
                     [-y, x,  0]], dtype=np.float64)

def rotation_matrix_a_to_b(a, b, eps=1e-12):
    """
    벡터 a를 b로 회전시키는 3x3 회전행렬(Rodrigues).
    a, b는 3D 벡터.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < eps or b_norm < eps:
        return np.eye(3, dtype=np.float64)

    a = a / a_norm
    b = b / b_norm

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)

    if s < eps:
        # 평행(같은 방향) or 반평행(반대 방향)
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        # 반대 방향(180도 회전): a와 수직인 축을 임의 선택
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis)
        v = v / (np.linalg.norm(v) + eps)
        K = _skew(v)
        return np.eye(3) + 2.0 * (K @ K)  # θ=π → R = I + 2K^2

    K = _skew(v)
    R = np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))
    return R

def pca_numpy(points_centered):
    """
    중심이 0인 points(N,3)에 대해 3x3 공분산의 고유분해.
    반환: (eigs(길이3), evecs(3x3; 열벡터가 고유벡터))
    """
    cov = np.cov(points_centered.T)
    S, U = np.linalg.eigh(cov)  # 오름차순 고유값
    return S, U

# ---------- 정규화 코어 ----------

def normalize_points_minEV_to_X(points, target_max=1.0, place_x='min', eps=1e-12):
    """
    1) 중심이동 → 2) 최소분산축을 +X로 회전 → 3) 최대치수로 나눠 정규화(최대변=1)
    4) target_max 배 스케일 → 5) X=0(yz면) 배치(place_x: min|center|max)

    반환: (pts_out, tf_dict)
      - tf_dict: center, R, scale_norm, target_max, place_x, dx
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points는 (N,3)이어야 합니다.")

    # 1) 중심 이동
    center = pts.mean(axis=0, keepdims=True)
    p0 = pts - center

    # 2) PCA → 최소분산축을 X로
    S, U = pca_numpy(p0)
    smallest_ev = U[:, np.argmin(S)]  # (3,)
    R = rotation_matrix_a_to_b(smallest_ev, np.array([1.0, 0.0, 0.0]))
    p1 = (R @ p0.T).T  # (N,3)

    # 3) 최대 치수로 나눠 정규화 (max extent = 1)
    pmin = p1.min(axis=0)
    pmax = p1.max(axis=0)
    extents = pmax - pmin
    scale_norm = float(np.max(extents))
    if not np.isfinite(scale_norm) or scale_norm <= eps:
        raise ValueError("유효한 스케일을 계산하지 못했습니다. (extents가 0에 가깝습니다)")

    p2 = p1 / scale_norm

    # 4) 원하는 최종 크기
    p3 = p2 * float(target_max)

    # 5) X=0 면 배치
    if place_x == 'min':
        dx = -float(p3[:, 0].min())
    elif place_x == 'center':
        dx = -float(p3[:, 0].mean())
    elif place_x == 'max':
        dx = -float(p3[:, 0].max())
    else:
        raise ValueError("place_x는 'min' | 'center' | 'max' 중 하나여야 합니다.")

    p4 = p3.copy()
    p4[:, 0] += dx

    tf = {
        "center": center.reshape(-1).tolist(),
        "rotation": R.tolist(),            # 원본→정렬로 가는 R
        "scale_norm": scale_norm,          # max extent
        "target_max": float(target_max),
        "place_x": place_x,
        "dx": dx                           # 마지막 X 평행이동
    }
    return p4, tf

# ---------- 파일 I/O 파이프라인 ----------

def run_normalize_ply(path_in, path_out=None, target_max=1.0, unit="unitless",
                      place_x='min', recompute_normals=True, save_tf_json=True):
    mesh = o3d.io.read_triangle_mesh(path_in)
    if not mesh.has_vertices():
        raise ValueError("메시에 유효한 vertex 정보가 없습니다.")

    pts = np.asarray(mesh.vertices, dtype=np.float64)
    pts_out, tf = normalize_points_minEV_to_X(
        pts, target_max=target_max, place_x=place_x
    )

    mesh.vertices = o3d.utility.Vector3dVector(pts_out)
    if recompute_normals:
        try:
            mesh.compute_vertex_normals()
        except Exception:
            pass

    if path_out is None:
        base, _ = os.path.splitext(path_in)
        suffix = f"_norm_minEVx_x0-{place_x}"
        if unit != "unitless":
            suffix += f"_{target_max}{unit}"
        else:
            if float(target_max) != 1.0:
                suffix += f"_{target_max}"
        path_out = base + suffix + ".ply"

    ok = o3d.io.write_triangle_mesh(path_out, mesh, write_ascii=False)
    if not ok:
        raise IOError(f"PLY 저장 실패: {path_out}")

    print(f"[✓] 저장 완료: {path_out}")
    print(f"    - target_max={target_max} ({unit}), place_x={place_x}")
    print(f"    - scale_norm(max extent)={tf['scale_norm']:.6g}, dx={tf['dx']:.6g}")

    # 변환 파라미터 저장(옵션)
    json_path = None
    if save_tf_json:
        json_path = path_out.replace(".ply", "_tf.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(tf, f, ensure_ascii=False, indent=2)
        print(f"[i] 변환 파라미터 저장: {json_path}")

    return path_out, json_path

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        description="PLY 정규화(최소분산축→X) + X=0(yz면) 배치"
    )
    ap.add_argument("--path_in", required=True, help="입력 PLY 경로")
    ap.add_argument("--path_out", help="출력 PLY 경로 (선택)")
    ap.add_argument("--target_max", type=float, default=1.0,
                    help="정규화 후 최종 최대 치수 (기본=1.0)")
    ap.add_argument("--unit", choices=["unitless", "mm", "cm", "m"],
                    default="unitless", help="정보성 단위(파일명 접미사용)")
    ap.add_argument("--place_x", choices=["min", "center", "max"], default="min",
                    help="X=0 면 배치 기준 (min: minX→0 / center: meanX→0 / max: maxX→0)")
    ap.add_argument("--no_normals", action="store_true", help="노멀 재계산 끄기")
    ap.add_argument("--no_tfjson", action="store_true", help="변환 파라미터 JSON 저장 끄기")

    args = ap.parse_args()

    run_normalize_ply(
        args.path_in,
        path_out=args.path_out,
        target_max=args.target_max,
        unit=args.unit,
        place_x=args.place_x,
        recompute_normals=(not args.no_normals),
        save_tf_json=(not args.no_tfjson),
    )

if __name__ == "__main__":
    main()
