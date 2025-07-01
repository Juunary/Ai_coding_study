import numpy as np
import argparse
import os

def shift_x_value(path_in, shift):
    try:
        data = np.loadtxt(path_in)
    except Exception as e:
        print(f"[오류] 파일 읽기 실패: {e}")
        return

    if data.shape[1] not in [3, 4]:
        print(f"[오류] 입력 파일은 .xyz 또는 .xyzc 형식이어야 합니다. 현재 shape: {data.shape}")
        return

    # X좌표(0번째 열) shift만큼 증가
    data[:, 0] += float(shift)

    # 저장 경로 설정
    suffix = f"_xShift{shift}".replace(".", "p")
    ext = ".xyzc" if data.shape[1] == 4 else ".xyz"
    path_out = path_in.replace(ext, f"{suffix}{ext}")

    # 과학적 표기법으로 저장
    fmt = ("%.18e " * data.shape[1]).strip()
    np.savetxt(path_out, data, fmt=fmt)
    print(f"[완료] X축 {shift}만큼 이동된 파일 저장됨: {path_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X축 좌표 값을 입력한 만큼 증가 (.xyz/.xyzc)")
    parser.add_argument("path_in", type=str, help="입력 파일 경로")
    parser.add_argument("shift", type=float, help="X축 이동값 (예: 0.1)")
    args = parser.parse_args()

    shift_x_value(args.path_in, args.shift)
