import numpy as np
import argparse
import os

def convert_xyz_to_xyzc(path_in, label):
    # .xyz 파일 불러오기
    try:
        xyz = np.loadtxt(path_in)
    except Exception as e:
        print(f"[오류] 파일 불러오기 실패: {e}")
        return

    if xyz.shape[1] != 3:
        print(f"[오류] 입력 파일은 3차원 좌표 형식이어야 합니다. 현재 shape: {xyz.shape}")
        return

    # 정수 라벨 추가
    label_column = np.full((xyz.shape[0], 1), int(label))
    xyzc = np.hstack((xyz, label_column))

    # 저장 경로 설정
    path_out = path_in.replace(".xyz", f"_label{label}.xyzc")
    np.savetxt(path_out, xyzc, fmt="%.6f %.6f %.6f %d")
    print(f"[완료] 저장됨: {path_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XYZ 파일에 정수 라벨 추가")
    parser.add_argument("path_in", type=str, help="입력 .xyz 파일 경로")
    parser.add_argument("label", type=int, help="추가할 정수 라벨 (예: 0, 1, 2...)")
    args = parser.parse_args()

    convert_xyz_to_xyzc(args.path_in, args.label)
