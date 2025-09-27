# ransac_segment_iter.py
import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud("input.ply")
pts = np.asarray(pcd.points)
labels = np.full(len(pts), -1, dtype=np.int64)
next_label = 0
remaining = pcd

# extract up to 10 planes (tune)
for i in range(10):
    plane_model, inliers = remaining.segment_plane(distance_threshold=0.003, ransac_n=3, num_iterations=1000)
    if len(inliers) < 200: break
    labels_indices = np.asarray(remaining.points)[inliers]
    # map indices back to original points: easiest if you operate on numpy and mask
    # Here we do mask method:
    mask = np.zeros(len(pts), dtype=bool)
    # find indices (this method requires tracking; for MVP prefer using numpy indexing)
    # Simplified: remove inliers from remaining and assign label
    labels[np.isin(pts, np.asarray(remaining.points)[inliers]).all(axis=1)] = next_label
    # remove inliers
    remaining_points = np.asarray(remaining.points)
    remaining = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.delete(remaining_points, inliers, axis=0)))
    next_label += 1

# Remaining points -> one label
labels[labels==-1] = next_label
np.save("labels.npy", labels)
