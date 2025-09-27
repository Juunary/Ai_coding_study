# python points_from_file.py impeller.ply points.npy
import numpy as np
import os, sys

def save_points_from_ply(in_path, out_npy):
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(in_path)
        pts = np.asarray(pcd.points, dtype=np.float32)
    except Exception:
        # fallback for simple XYZ/CSV: load numeric columns
        pts = np.loadtxt(in_path, delimiter=None, dtype=np.float32)
        if pts.ndim==1 and pts.size%3==0:
            pts = pts.reshape(-1,3)
    np.save(out_npy, pts)
    print("Saved", out_npy, "shape=", pts.shape, "dtype=", pts.dtype)

if __name__=="__main__":
    # Example usage:
    # python save_points_from_file.py input.ply points.npy
    import sys
    if len(sys.argv)<3:
        print("Usage: python save_points_from_file.py <in.ply|.xyz|.txt|.csv> <out_points.npy>")
        sys.exit(1)
    save_points_from_ply(sys.argv[1], sys.argv[2])
