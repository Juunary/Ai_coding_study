# cd C:\Users\user\Documents\GitHub\Ai_coding_study\TopoConstrained-NeuroBrep-MVP
# python -m tests.smoke_minimal

from __future__ import annotations
import os, numpy as np, json, tempfile, torch
from topocn.eval.bench_runner import run_impeller
from dataset.impeller_synth import make_synthetic_impeller

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if __name__=="__main__":
    # Prepare synthetic data dir
    tmp = tempfile.mkdtemp()
    pts, labels, types = make_synthetic_impeller(n_blades=3, n_per_blade=200, noise=0.003)
    np.save(os.path.join(tmp, "points.npy"), pts)
    np.save(os.path.join(tmp, "labels.npy"), labels)
    np.save(os.path.join(tmp, "types.npy"), types)
    out = os.path.join(tmp, "out")
    os.makedirs(out, exist_ok=True)
    cfg = os.path.join(os.path.dirname(__file__), "..", "configs", "impeller.yaml")
    cfg = os.path.abspath(cfg)
    run_impeller(tmp, cfg, out)
    print("Output dir:", out)
    print("Contents:", os.listdir(out))
    with open(os.path.join(out, "report.json"), "r") as f:
        print("Report:", json.load(f))
