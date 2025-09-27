# patch_solver_weights.py
from pathlib import Path

content = """from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import torch, numpy as np, random
from ..topo.graph import build_adjacency, Graph
from ..topo.constraints import annotate_edges
from ..topo.snap import greedy_snap
from .energy import total_energy, Weights
from ..geometry.fitters import initial_fit
from ..utils import log

@dataclass
class SolveConfig:
    seed: int = 0
    iters: int = 50
    lr: float = 1e-2
    snap_every: int = 10
    relabel_every: int = 10
    max_snap_edges: int = 5
    device: str = "cpu"
    weights: Weights = field(default_factory=Weights)

def set_seed(seed:int):
    import os, torch, numpy as np, random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def prepare_samples(points: np.ndarray, labels: np.ndarray, types: np.ndarray, patches: Dict[int, object]) -> Dict[int, torch.Tensor]:
    \"\"\"Prepare per-patch samples for data term. For INR, also return off-surface samples.\"\"\"
    samples = {}
    for k, p in patches.items():
        Pk = torch.from_numpy(points[labels==k].astype(np.float32))
        if hasattr(p, "kind") and p.kind==\"inr\":
            noise = 0.01*torch.randn_like(Pk)
            samples[k] = (Pk, Pk + noise)
        else:
            samples[k] = Pk
    return samples

def solve(points: np.ndarray, labels: np.ndarray, types: np.ndarray, cfg: SolveConfig) -> Tuple[Dict[int,object], Graph, Dict[str,float]]:
    # Ensure cfg.weights is a Weights instance (YAML/JSON will load to dict)
    if isinstance(cfg.weights, dict):
        cfg.weights = Weights(**cfg.weights)

    set_seed(cfg.seed)

    patches = initial_fit(points, labels, types)
    graph = build_adjacency(points, labels, k=12, max_pairs=1024)
    graph = annotate_edges(graph, patches)
    opt_params = []
    for p in patches.values():
        for name, val in p.__dict__.items():
            if isinstance(val, torch.nn.Parameter):
                opt_params.append(val)
        if hasattr(p, "model"):
            opt_params += list(p.model.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=cfg.lr)

    samples = prepare_samples(points, labels, types, patches)

    for it in range(1, cfg.iters+1):
        optimizer.zero_grad()
        E, scal = total_energy(graph, patches, samples, cfg.weights)
        E.backward()
        optimizer.step()
        if it % cfg.snap_every == 0:
            greedy_snap(graph, patches, w=dict(G1=1.0, G2=0.2, GAP=50.0), max_edges=cfg.max_snap_edges, alpha=0.5)
        if it % cfg.relabel_every == 0:
            graph = annotate_edges(graph, patches)
        if it % 5 == 0 or it==1:
            log.info(f\"Iter {it:03d} :: \" + \", \".join([f\"{k}={v:.4f}\" for k,v in scal.items()]))
    return patches, graph, scal
"""

p = Path("topocn/optimize/solver.py")
p.write_text(content, encoding="utf-8")
print("Patched", p.resolve())
