import open3d as o3d
import numpy as np
import argparse

def load_and_process_ply(path, scale=100.0, shift_x=0.0, rotate_deg_x=0.0):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()

    mesh.scale(scale, center=(0, 0, 0))

    mesh.translate((shift_x, 0, 0))

    if rotate_deg_x != 0.0:
        radians = np.deg2rad(rotate_deg_x)
        R = mesh.get_rotation_matrix_from_axis_angle([radians, 0, 0])
        mesh.rotate(R, center=(0, 0, 0))

    return mesh

def compute_area(mesh):
    return mesh.get_surface_area()

def compute_rmse_distance(mesh1, mesh2):
    pcd1 = mesh1.sample_points_uniformly(number_of_points=10000)
    pcd2 = mesh2.sample_points_uniformly(number_of_points=10000)

    dists1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    dists2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))

    rmse = np.sqrt((np.mean(dists1 ** 2) + np.mean(dists2 ** 2)) / 2)
    return rmse

def main(path1, path2):
    mesh1 = load_and_process_ply(path1, scale=100.0, shift_x=6.95, rotate_deg_x=11.8)
    mesh2 = load_and_process_ply(path2, scale=100.0)

    area1 = compute_area(mesh1)
    area2 = compute_area(mesh2)

    rmse = compute_rmse_distance(mesh1, mesh2)

    print(f"Regenerate ver: {area1:.3f} mm², Origin ver: {area2:.3f} mm²")
    print(f"Diff (RMSE): {rmse:.4f} mm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="두 PLY 파일의 면적 및 형상 비교")
    parser.add_argument("--path1", required=True, help="첫 번째 PLY 파일")
    parser.add_argument("--path2", required=True, help="두 번째 PLY 파일")
    args = parser.parse_args()

    main(args.path1, args.path2)
