import numpy as np, open3d as o3d, sys
pts = np.load("assets/points.npy")
labels = np.load("assets/labels.npy")
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
# color map by label
maxL = int(labels.max())+1
rng = __import__("numpy").random.RandomState(0)
colors = (rng.rand(maxL,3) * 0.9 + 0.1)[labels]
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])