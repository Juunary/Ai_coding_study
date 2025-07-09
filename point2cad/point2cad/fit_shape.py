import argparse
import numpy as np
import torch
from fitting_one_surface import fit_basic_primitives

def main():
    parser = argparse.ArgumentParser(description="xyzc 파일로부터 best primitive fit 타입 출력 + 임펠러 예측")
    parser.add_argument('--path_in', required=True, help=".xyzc 입력 파일 경로")
    args = parser.parse_args()

    # 1. 점군 로드
    data = np.loadtxt(args.path_in)
    pts = torch.from_numpy(data[:, :3]).float()

    # 2. Primitive fitting
    shapes = fit_basic_primitives(pts)

    # 3. 에러 정리
    errs = {
        'sphere'  : float(shapes['sphere_err']),
        'cone'    : float(shapes['cone_err']),
        'cylinder': float(shapes['cylinder_err']),
        'plane'   : float(shapes['plane_err'])
    }

    # 4. 최적 primitive 출력
    best = min(errs, key=errs.get)
    print(f"\n=== Fit Result for {args.path_in} ===")
    print(f"➤ Best fit primitive : {best}\n")
    for t, e in errs.items():
        print(f"  {t:8s} error : {e:.10f}")

    # 5. 임펠러 구조 예측 조건
    e = errs  # 간단히 변수 축약
    if e['sphere'] < e['cone'] < e['cylinder'] < e['plane']:
        print("\n✅ 구조 예측: 이 점군은 임펠러 구조일 가능성이 높습니다.")
    else:
        print("\n❌ 구조 예측: 임펠러 형상이라고 보기는 어렵습니다.")

if __name__ == '__main__':
    main()
