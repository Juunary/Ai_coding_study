from __future__ import annotations
from typing import Dict, Tuple, List
import torch
from .graph import Graph
from ..geometry.primitives import PlanePatch, CylinderPatch, SpherePatch, ConePatch
from ..utils import log

Tensor = torch.Tensor

def _violation_score(labels: dict, w: dict) -> float:
    return (w.get("G1", 1.0) * labels.get("G1", 0.0) 
           + w.get("G2", 0.2) * labels.get("G2", 0.0) 
           + w.get("GAP", 0.5) * labels.get("GAP", 0.0))

def greedy_snap(graph: Graph, patches: Dict[int, object], w: dict, max_edges: int = 10, alpha: float = 0.3, snap_every: int = 5):
    """Greedy projection to reduce large violations.
    alpha controls how strongly parameters move toward the snapped configuration (0..1).
    """
    # rank edges by violation
    scored = sorted(graph.edges.items(), key=lambda kv: _violation_score(kv[1].labels, w), reverse=True)
    acted = 0
    for (i, j), ed in scored:
        if acted >= max_edges:
            break
        pi, pj = ed.pairs_i, ed.pairs_j
        # Case 1: plane-plane nearly coplanar/parallel -> co-plane snap
        a = patches[i]; b = patches[j]
        if isinstance(a, PlanePatch) and isinstance(b, PlanePatch):
            n = (a.n / (a.n.norm() + 1e-12) + b.n / (b.n.norm() + 1e-12))
            n = n / (n.norm() + 1e-12)
            d = (a.d + b.d) / 2.0
            # apply small move
            a.n.data = (1 - alpha) * a.n.data + alpha * n.data
            b.n.data = (1 - alpha) * b.n.data + alpha * n.data
            a.d.data = (1 - alpha) * a.d.data + alpha * d.data
            b.d.data = (1 - alpha) * b.d.data + alpha * d.data
            acted += 1
            continue
        # Case 2: cylinder-cylinder coax -> align axes & radius
        if isinstance(a, CylinderPatch) and isinstance(b, CylinderPatch):
            ua = a.a / (a.a.norm() + 1e-12); ub = b.a / (b.a.norm() + 1e-12)
            u = (ua + ub); u = u / (u.norm() + 1e-12)
            ra = a.r.abs(); rb = b.r.abs(); r = (ra + rb) / 2.0
            a.a.data = (1 - alpha) * a.a.data + alpha * u.data
            b.a.data = (1 - alpha) * b.a.data + alpha * u.data
            a.r.data = (1 - alpha) * a.r.data + alpha * r.data
            b.r.data = (1 - alpha) * b.r.data + alpha * r.data
            # centers: project means to common axis
            mi = pi.mean(dim=0); mj = pj.mean(dim=0); m = (mi + mj) / 2.0
            # set c near m projected to axis
            a.c.data = (1 - alpha) * a.c.data + alpha * (m - (m @ u) * u).data
            b.c.data = (1 - alpha) * b.c.data + alpha * (m - (m @ u) * u).data
            acted += 1
            continue
        # Case 3: plane-cylinder angle constraint -> make axis ⟂ plane
        if (isinstance(a, PlanePatch) and isinstance(b, CylinderPatch)) or (isinstance(b, PlanePatch) and isinstance(a, CylinderPatch)):
            if isinstance(a, PlanePatch): pl, cy = a, b
            else: pl, cy = b, a
            n = pl.n / (pl.n.norm() + 1e-12)
            u = cy.a - (cy.a @ n) * n  # remove normal component for parallel snap
            if u.norm() < 1e-6:
                u = torch.randn_like(n)  # fallback
            u = u / (u.norm() + 1e-12)
            # choose orth or parallel based on which reduces loss more (heuristic: use labels ANG if available)
            # here, snap to orthogonal
            cy.a.data = (1 - alpha) * cy.a.data + alpha * (n - (n @ u) * u).data
            acted += 1
            continue
        # INR-related snap would adjust training boundary points; handled in optimizer by resampling.
    if acted > 0:
        log.info(f"Snap applied on {acted} edges (alpha={alpha})")
