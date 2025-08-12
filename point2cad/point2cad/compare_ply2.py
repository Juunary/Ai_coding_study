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

def compute_rmse_distance(mesh1, mesh2):
    pcd1 = mesh1.sample_points_uniformly(number_of_points=10000)
    pcd2 = mesh2.sample_points_uniformly(number_of_points=10000)

    dists1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    dists2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))

    rmse = np.sqrt((np.mean(dists1 ** 2) + np.mean(dists2 ** 2)) / 2)
    return rmse

def compute_bounding_box_diff(mesh1, mesh2):
    aabb1 = mesh1.get_axis_aligned_bounding_box()
    aabb2 = mesh2.get_axis_aligned_bounding_box()

    size1 = aabb1.get_extent()
    size2 = aabb2.get_extent()

    diff = np.abs(size1 - size2)
    return size1, size2, diff

def compute_geometric_accuracy(mesh1, mesh2):
    pcd1 = mesh1.sample_points_uniformly(number_of_points=10000)
    pcd2 = mesh2.sample_points_uniformly(number_of_points=10000)

    dists1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    dists2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))

    chamfer = (np.mean(dists1) + np.mean(dists2)) / 2
    hausdorff = max(np.max(dists1), np.max(dists2))

    return chamfer, hausdorff

def compute_surface_matching_accuracy(source_mesh, target_mesh):
    pcd_source = source_mesh.sample_points_uniformly(number_of_points=10000)
    pcd_target = target_mesh.sample_points_uniformly(number_of_points=10000)

    dists = np.asarray(pcd_source.compute_point_cloud_distance(pcd_target))
    mean_error = np.mean(dists)
    max_error = np.max(dists)
    return mean_error, max_error


def print_evaluation(area1, area2, rmse, bbox1, bbox2, bbox_diff, chamfer, hausdorff, mean_surface, max_surface):
    print("\n===== 비교 결과 =====")
    print(f" 면적: Regenerated = {area1:.3f} mm² / Origin = {area2:.3f} mm²")
    print(f" 절대 위치 정확도 (RMSE): {rmse:.4f} mm")
    
    print(f"\n 크기 비교 (가로/세로/높이):")
    print(f"Regenerated Size (X, Y, Z): {bbox1[0]:.3f}, {bbox1[1]:.3f}, {bbox1[2]:.3f}")
    print(f"Original Size    (X, Y, Z): {bbox2[0]:.3f}, {bbox2[1]:.3f}, {bbox2[2]:.3f}")
    print(f"차이 (절대값 mm):            {bbox_diff[0]:.4f}, {bbox_diff[1]:.4f}, {bbox_diff[2]:.4f}")

    threshold = 0.05
    result = ["o" if d <= threshold else "❌" for d in bbox_diff]
    print(f"정밀도 기준(±{threshold}mm) 만족 여부:        {result[0]} {result[1]} {result[2]}")

    print(f"\n 기하학적 정확도:")
    print(f"Chamfer Distance   : {chamfer:.4f} mm → {'o 기준 만족' if chamfer <= 0.1 else '❌ 기준 초과'}")
    print(f"Hausdorff Distance : {hausdorff:.4f} mm → {'o 기준 만족' if hausdorff <= 0.2 else '❌ 기준 초과'}")
    
    print(f"\n 표면 일치도:")
    #print(f"평균 오차: {mean_surface:.4f} mm → {'✅' if mean_surface <= 0.01 else '❌'}")
    print(f"최대 오차: {max_surface:.4f} mm → {'o' if max_surface <= 0.1 else '❌'}")

def main(path1, path2):
    mesh1 = load_and_process_ply(path1, scale=100.0, shift_x=6.95, rotate_deg_x=11.8)
    mesh2 = load_and_process_ply(path2, scale=100.0)

    area1 = mesh1.get_surface_area()
    area2 = mesh2.get_surface_area()

    rmse = compute_rmse_distance(mesh1, mesh2)
    chamfer, hausdorff = compute_geometric_accuracy(mesh1, mesh2)
    bbox1, bbox2, bbox_diff = compute_bounding_box_diff(mesh1, mesh2)
    mean_surface, max_surface = compute_surface_matching_accuracy(mesh1, mesh2)

    # 스케일 복원
    area1 /= 10000
    area2 /= 10000
    rmse /= 100
    chamfer /= 100
    hausdorff /= 100
    bbox1 /= 100
    bbox2 /= 100
    bbox_diff /= 100
    mean_surface /= 100
    max_surface /= 100

    print_evaluation(area1, area2, rmse, bbox1, bbox2, bbox_diff, chamfer, hausdorff, mean_surface, max_surface)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PLY 형상 비교 및 정밀도 평가")
    parser.add_argument("--path1", required=True, help="복원된 PLY 파일 경로")
    parser.add_argument("--path2", required=True, help="원본 PLY 파일 경로")
    args = parser.parse_args()

    main(args.path1, args.path2)
