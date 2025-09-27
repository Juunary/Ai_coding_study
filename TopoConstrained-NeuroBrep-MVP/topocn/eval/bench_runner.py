
from __future__ import annotations
import os, json, yaml, numpy as np, torch
from ..optimize.solver import solve, SolveConfig
from ..export.brep import assemble_brep
from ..export.step_writer import write_step
from ..eval.metrics_cad import count_g1_discontinuities, step_success
from ..eval.metrics_geom import chamfer, hausdorff
from ..topo.graph import Graph
from ..utils import log

def run_impeller(data_dir: str, config_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(config_path, "r") as f:
        cfg_yaml = yaml.safe_load(f)
    pts = np.load(os.path.join(data_dir, "points.npy")).astype(np.float32)
    labels = np.load(os.path.join(data_dir, "labels.npy")).astype(np.int64)
    types = np.load(os.path.join(data_dir, "types.npy")).astype(np.int64)
    cfg = SolveConfig(**cfg_yaml.get("solver", {}))
    patches, graph, scal = solve(pts, labels, types, cfg)

    # export STEP if possible
    shape = assemble_brep(patches, graph)
    step_path = os.path.join(out_dir, "result.step")
    step_ok = False
    if shape is not None:
        step_ok = write_step(shape, step_path)

    
# metrics
from ..topo.validators import gap_violations, self_intersection_soft

    # Build edge label map for G1
    edge_labels = {(i,j): ed.labels for (i,j), ed in graph.edges.items()}
    g1_disc = count_g1_discontinuities(edge_labels, tau_deg=5.0)
    # geometric metrics vs. fitted projections (proxy)
    P = torch.from_numpy(pts)
    Pfit = []
    for pid, patch in patches.items():
        Pi = P[labels==pid]
        if hasattr(patch, "project"):
            Pfit.append(patch.project(Pi))
        else:
            Pfit.append(Pi)
    Pfit = torch.cat(Pfit, dim=0)
    CD = chamfer(P, Pfit); HD = hausdorff(P, Pfit)

    gap_count = 0; self_soft = 0.0
for (i,j), ed in graph.edges.items():
    gap_count += gap_violations(ed.pairs_i, ed.pairs_j, eps=1e-3)
    self_soft += self_intersection_soft(ed.pairs_i, ed.pairs_j, tau=1e-3)
report = dict(E=scal, G1_discont=g1_disc, GAP_violations=gap_count,
              SELF_soft=self_soft, STEP_success=bool(step_ok), Chamfer=CD, Hausdorff=HD)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report: {report}")
    if shape is None:
        log.warn("STEP not produced (pythonocc-core missing). You can still inspect report.json.")

