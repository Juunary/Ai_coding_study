import numpy as np
import argparse
import os

def flip_x(path_in):
    try:
        data = np.loadtxt(path_in)
    except Exception as e:
        print(f"[오류] 파일 읽기 실패: {e}")
        return

    if data.shape[1] not in [3, 4]:
        print(f"[오류] .xyz 또는 .xyzc 형식만 지원합니다. 현재 shape: {data.shape}")
        return

    # X축 반전
    data[:, 0] *= -1

    # 출력 경로 설정
    suffix = "_flippedX"
    ext = ".xyzc" if data.shape[1] == 4 else ".xyz"
    path_out = path_in.replace(ext, f"{suffix}{ext}")

    # 저장 포맷
    fmt = "%.18e " * data.shape[1]
    np.savetxt(path_out, data, fmt=fmt.strip())
    print(f"[완료] X축 반전 파일 저장됨: {path_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X축 기준 반전 (.xyz 또는 .xyzc)")
    parser.add_argument("path_in", type=str, help="입력 파일 경로")
    args = parser.parse_args()

    flip_x(args.path_in)
