"""Infer the surface type of each cluster in a point cloud via the shape vector DB.

    python infer_shape.py --path_in ../assets/abc_00949.xyzc
    python infer_shape.py --path_in patch.xyz            # no label column: one cluster

Input format follows main.py: x y z [label] per line.
"""
import argparse
import os

import numpy as np

from shape_encoder import ShapeEncoder
from shape_rag import ShapeVectorDB, infer_shape_type

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path_in", required=True)
    ap.add_argument("--encoder", default=os.path.join(HERE, "artifacts", "shape_encoder.npz"))
    ap.add_argument("--db", default=os.path.join(HERE, "artifacts", "shape_db.npz"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min_points", type=int, default=20)
    ap.add_argument("--min_similarity", type=float, default=0.8,
                    help="below this nearest-example similarity the cluster is reported as unknown")
    args = ap.parse_args()

    enc = ShapeEncoder.load(args.encoder)
    db = ShapeVectorDB.load(args.db)
    data = np.loadtxt(args.path_in).astype(np.float64)
    labels = data[:, 3].astype(int) if data.shape[1] >= 4 else np.zeros(len(data), int)

    print(f"{'label':>5s} {'points':>6s}  {'type':9s} {'conf':>5s} {'sim':>5s}  votes")
    for lab in np.unique(labels):
        pts = data[labels == lab, :3]
        if len(pts) < args.min_points:
            continue
        r = infer_shape_type(pts, enc, db, k=args.k, min_similarity=args.min_similarity)
        votes = ", ".join(f"{t}:{s:.2f}" for t, s in r["votes"].items())
        print(f"{lab:5d} {len(pts):6d}  {r['type']:9s} {r['confidence']:5.2f} {r['similarity']:5.2f}  {votes}")


if __name__ == "__main__":
    main()
