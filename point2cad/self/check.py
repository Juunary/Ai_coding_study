import open3d as o3d

# 1) 포인트 클라우드 읽어오기
pcd_scan = o3d.io.read_point_cloud(r"C:\Users\user\Documents\GitHub\Ai_coding_study\point2cad\out_Data\origin\mona.ply")
pcd_p2cad = o3d.io.read_point_cloud(r"C:\Users\user\Documents\GitHub\Ai_coding_study\point2cad\out_Data\clipped\mesh_mona.ply")

# 2) 제대로 불러왔는지, 점 개수 등을 확인
print("Scan point cloud has", len(pcd_scan.points), "points")
print("P2CAD point cloud has", len(pcd_p2cad.points), "points")

# 3) 시각적으로 확인 (optional)
o3d.visualization.draw_geometries([ pcd_p2cad])
o3d.visualization.draw_geometries([ pcd_scan])