
from __future__ import annotations
from typing import Dict, Tuple
import torch
from .graph import Graph, EdgeData
from ..geometry.primitives import PlanePatch, CylinderPatch, SpherePatch, ConePatch
from ..utils import log

Tensor = torch.Tensor

def _get_normal(patch, x: Tensor) -> Tensor:
    return patch.normal(x)

def _principal_curvatures(patch, x: Tensor) -> Tuple[Tensor, Tensor]:
    if hasattr(patch, "principal_curvatures"):
        return patch.principal_curvatures(x)
    # INR wrapper with method
    return patch.principal_curvatures(x)

def annotate_edges(graph: Graph, patches: Dict[int, object]) -> Graph:
    """Compute edge labels/metrics (G1, G2, coaxial/parallel/orth, gap)."""
    for (i,j), ed in graph.edges.items():
        pi, pj = ed.pairs_i, ed.pairs_j
        # G1: tangent continuity via normals
        ni = _get_normal(patches[i], pi)
        nj = _get_normal(patches[j], pj)
        cosang = (ni * nj).sum(dim=1).abs().clamp(0,1)
        g1 = 1.0 - cosang.mean()
        # G2: curvature diff (use principal magnitudes)
        k1i, k2i = _principal_curvatures(patches[i], pi)
        k1j, k2j = _principal_curvatures(patches[j], pj)
        # match by sorted magnitude
        ki = torch.sort(torch.stack([k1i.abs(), k2i.abs()], dim=1), dim=1).values
        kj = torch.sort(torch.stack([k1j.abs(), k2j.abs()], dim=1), dim=1).values
        g2 = (ki - kj).pow(2).mean()
        # gap
        gap = (pi - pj).norm(dim=1).mean()
        # angular relations using characteristic directions or normals
        ang = 0.0
        coax = 0.0
        ui = getattr(patches[i], "characteristic_direction", lambda: None)()
        uj = getattr(patches[j], "characteristic_direction", lambda: None)()
        if ui is not None and uj is not None:
            ui = ui / (ui.norm()+1e-12); uj = uj / (uj.norm()+1e-12)
            ang = 1.0 - (ui @ uj).abs().item()
            # coaxial: also measure minimal axis distance from two points (use means)
            ci = pi.mean(dim=0); cj = pj.mean(dim=0)
            v = cj - ci
            d_ax = (v - (v @ ui)*ui).norm().item()
            coax = ang + d_ax  # lower is better; stored as "coax_score"
        # store
        ed.labels = dict(G1=float(g1.detach()),
                         G2=float(g2.detach()),
                         GAP=float(gap.detach()),
                         ANG=float(ang),
                         COAX=float(coax))
    return graph
