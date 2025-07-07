import argparse
import numpy as np
import torch
from fitting_one_surface import fit_basic_primitives

def main():
    parser = argparse.ArgumentParser(description="xyzc 파일로부터 best primitive fit 타입 출력")
    parser.add_argument('--path_in', required=True, help=".xyzc 입력 파일 경로")
    args = parser.parse_args()

    data = np.loadtxt(args.path_in)
    pts = torch.from_numpy(data[:, :3]).float()

    shapes = fit_basic_primitives(pts)

    # .item() 제거
    errs = {
        'plane'   : float(shapes['plane_err']),
        'sphere'  : float(shapes['sphere_err']),
        'cylinder': float(shapes['cylinder_err']),
        'cone'    : float(shapes['cone_err'])
    }

    best = min(errs, key=errs.get)

    print(f"\n=== Fit Result for {args.path_in} ===")
    print(f"➤ Best fit primitive : {best}\n")
    for t, e in errs.items():
        print(f"  {t:8s} error : {e:.6f}")

if __name__ == '__main__':
    main()
