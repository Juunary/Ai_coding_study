import open3d as o3d
import numpy as np
import argparse
import os

def normalize_points(points):
    EPS = np.finfo(np.float32).eps
    # 중심 정렬
    points -= np.mean(points, axis=0, keepdims=True)
    # PCA 회전
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    rotation = eigvecs
    points = points @ rotation
    # 크기 정규화
    scale = np.max(points.max(axis=0) - points.min(axis=0)) + EPS
    points /= scale
    return points

def normalize_ply(path_in, path_out=None):
    mesh = o3d.io.read_triangle_mesh(path_in)
    if not mesh.has_vertices():
        raise ValueError("메시에 유효한 vertex 정보가 없습니다.")

    points = np.asarray(mesh.vertices)
    norm_points = normalize_points(points)
    mesh.vertices = o3d.utility.Vector3dVector(norm_points)

    if path_out is None:
        base, ext = os.path.splitext(path_in)
        path_out = base + "_normalized.ply"

    o3d.io.write_triangle_mesh(path_out, mesh)
    print(f"[✓] 정규화된 파일 저장 완료: {path_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLY 정규화 및 저장 스크립트")
    parser.add_argument("--path_in", required=True, help="입력 PLY 파일 경로")
    parser.add_argument("--path_out", help="출력 PLY 파일 경로 (선택)")
    args = parser.parse_args()

    normalize_ply(args.path_in, args.path_out)
