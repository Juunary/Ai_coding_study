#!/usr/bin/env python3
# print_casing_axis_scales.py
import os, argparse, numpy as np, open3d as o3d

def aabb_lengths_mm(mesh, scale=1.0):
    return mesh.get_axis_aligned_bounding_box().get_extent() / scale  # [Lx,Ly,Lz]

def parse_target(s):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("--target 은 'Lx,Ly,Lz' 형태여야 합니다.")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)

def mean_lengths_in_dir(path_dir, pattern, start, end, scale=1.0):
    vals = []
    for i in range(start, end+1):
        f = os.path.join(path_dir, pattern.format(i))
        if not os.path.exists(f):
            continue
        m = o3d.io.read_triangle_mesh(f)
        if m.is_empty():
            continue
        vals.append(aabb_lengths_mm(m, scale))
    if not vals:
        raise RuntimeError("테스트 폴더에서 유효한 PLY를 하나도 찾지 못했습니다.")
    V = np.vstack(vals)  # N x 3
    return V.mean(axis=0)

def main():
    ap = argparse.ArgumentParser(description="각 축별 스케일 계수 계산 (ref → target)")
    ap.add_argument("--ref", required=True, help="참조 casing.ply 경로")
    ap.add_argument("--test_dir", help="테스트 폴더 (평균 타깃 자동계산용)")
    ap.add_argument("--pattern", default="mesh_casing_{:03d}.ply")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end",   type=int, default=120)
    ap.add_argument("--scale", type=float, default=1.0, help="단위 스케일(둘 다 동일 단위면 1)")
    ap.add_argument("--target", type=str, default=None,
                    help="직접 타깃 치수(mm) 지정: 'Lx,Ly,Lz' (예: 0.580897,0.999532,0.999630)")
    args = ap.parse_args()

    # 1) 참조 치수
    ref = o3d.io.read_triangle_mesh(args.ref)
    if ref.is_empty():
        raise RuntimeError(f"빈 메쉬: {args.ref}")
    Lref = aabb_lengths_mm(ref, args.scale)

    # 2) 타깃 치수: --target 이 있으면 그대로, 없으면 테스트 폴더 평균
    if args.target is not None:
        Ltar = parse_target(args.target)
    else:
        if not args.test_dir:
            raise RuntimeError("--target 없으면 --test_dir 가 필요합니다.")
        Ltar = mean_lengths_in_dir(args.test_dir, args.pattern, args.start, args.end, args.scale)

    # 3) 축별 스케일 계수
    with np.errstate(divide='ignore', invalid='ignore'):
        S = Ltar / Lref
    if np.any(~np.isfinite(S)):
        raise RuntimeError(f"참조 치수에 0 또는 비정상 값이 포함: Lref={Lref}")

    sx, sy, sz = S.tolist()
    vol_scale = float(sx*sy*sz)
    vol_pct = (vol_scale - 1.0) * 100.0

    # 4) 출력 (소수점 4자리)
    print("=== Axis-aligned AABB lengths (mm) ===")
    print(f"Ref   Lx,Ly,Lz = {Lref[0]:.6f}, {Lref[1]:.6f}, {Lref[2]:.6f}")
    print(f"TargetLx,Ly,Lz = {Ltar[0]:.6f}, {Ltar[1]:.6f}, {Ltar[2]:.6f}")
    print("\n=== Scale factors to apply on ref → target (non-uniform) ===")
    print(f"x:{sx:.4f} y:{sy:.4f} z:{sz:.4f}")
    print(f"(volume scale ≈ {vol_scale:.6f}, Δvolume ≈ {vol_pct:+.3f}%)")

if __name__ == "__main__":
    main()
