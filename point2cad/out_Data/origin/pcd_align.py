import numpy as np
import open3d as o3d

# -----------------------------------------------------------
# 1) PLY 파일 불러오기
# -----------------------------------------------------------
scan_path = r"C:\Users\user\Documents\GitHub\Ai_coding_study\point2cad\out_Data\origin\mesh63.ply"
p2cad_path = r"C:\Users\user\Documents\GitHub\Ai_coding_study\point2cad\out_Data\origin\p2cmesh.ply"

pcd_scan = o3d.io.read_point_cloud(scan_path)
pcd_p2cad = o3d.io.read_point_cloud(p2cad_path)

print("[INFO] Scan point cloud:", len(pcd_scan.points), "points")
print("[INFO] P2CAD point cloud:", len(pcd_p2cad.points), "points")

# -----------------------------------------------------------
# 2) Y축 90도 회전 (수동 변환)
# -----------------------------------------------------------
theta = np.radians(90)
Ry_90 = np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0,             1,             0],
    [-np.sin(theta), 0, np.cos(theta)]
])
T_yrotate = np.eye(4)
T_yrotate[:3, :3] = Ry_90

# p2cad에 회전 적용
pcd_p2cad.rotate(Ry_90, center=pcd_p2cad.get_center())
# 또는 transform(T_yrotate) 방식도 동일:
# pcd_p2cad.transform(T_yrotate)

# -----------------------------------------------------------
# 3) 스케일 조정 (예: 720배)
# -----------------------------------------------------------
pcd_p2cad.scale(720, center=pcd_p2cad.get_center())

# -----------------------------------------------------------
# 4) AABB(축정렬 Bounding Box) 중심 일치 ("1번 방법")
# -----------------------------------------------------------
bb_scan = pcd_scan.get_axis_aligned_bounding_box()
bb_p2cad = pcd_p2cad.get_axis_aligned_bounding_box()

center_scan = bb_scan.get_center()      # 스캔 모델 중심
center_p2cad = bb_p2cad.get_center()    # P2CAD 모델 중심

shift = center_scan - center_p2cad

T_shift = np.eye(4)
T_shift[:3, 3] = shift

pcd_p2cad.transform(T_shift)  # 평행이동 적용

# -----------------------------------------------------------
# (옵션) 5) ICP로 최종 미세 정렬
# -----------------------------------------------------------
threshold = 50.0  # 스케일이 커졌으므로, 이전보다 큰 threshold가 필요할 수 있음
reg_icp = o3d.pipelines.registration.registration_icp(
    source=pcd_p2cad,
    target=pcd_scan,
    max_correspondence_distance=threshold,
    init=np.eye(4),  # 이미 수동정렬로 어느 정도 맞춘 상태이므로 단위행렬 init
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
print("[INFO] ICP fitness:", reg_icp.fitness, 
      "| ICP inlier_rmse:", reg_icp.inlier_rmse)
pcd_p2cad_icp = pcd_p2cad.transform(reg_icp.transformation)

# -----------------------------------------------------------
# 6) 시각화 및 결과 저장
# -----------------------------------------------------------
# 색상을 달리해서 겹침 상태 확인
pcd_scan.paint_uniform_color([1, 0, 0])          # 빨강
pcd_p2cad_icp.paint_uniform_color([0, 1, 0])     # 초록

o3d.visualization.draw_geometries([pcd_scan, pcd_p2cad_icp],
                                  window_name="Aligned Point Clouds")

aligned_path = "p2cmesh_aligned.ply"
o3d.io.write_point_cloud(aligned_path, pcd_p2cad_icp)
print("[INFO] Saved aligned point cloud to:", aligned_path)
