
from __future__ import annotations
import os, numpy as np

def load_impeller_dir(path: str):
    pts = np.load(os.path.join(path, "points.npy")).astype(np.float32)
    labels = np.load(os.path.join(path, "labels.npy")).astype(np.int64)
    types = np.load(os.path.join(path, "types.npy")).astype(np.int64)
    return pts, labels, types
