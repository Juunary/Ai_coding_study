
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
from ..topo.graph import Graph
from ..geometry.primitives import PlanePatch, CylinderPatch, SpherePatch, ConePatch
Tensor = torch.Tensor

def huber(x: Tensor, delta: float = 1.0) -> Tensor:
    absx = x.abs()
    return torch.where(absx < delta, 0.5 * x**2, delta * (absx - 0.5 * delta))

@dataclass
class Weights:
    w_G1: float = 1.0
    w_G2: float = 0.2
    w_coax: float = 0.5
    w_ang: float = 0.2
    w_gap: float = 50.0
    w_self: float = 0.1
    lambda_c: float = 1.0
    lambda_r: float = 1e-4

def E_data_patch(patch, P: Tensor) -> Tensor:
    if hasattr(patch, "sdf"): # all patches have
        if isinstance(P, tuple):
            X = P[0]
        else:
            X = P
        if hasattr(patch, "kind") and patch.kind=="inr":
            # INR data: signed value near 0 for on-surface; optionally eikonal
            y = patch.sdf(X)
            eik = 0.0
            # sample off-surface for eikonal if provided
            if len(P)==2:
                Xoff = P[1]
                Xmix = torch.cat([X, Xoff], dim=0).requires_grad_(True)
                ymix = patch.sdf(Xmix)
                g = torch.autograd.grad(ymix.sum(), Xmix, create_graph=True)[0]
                eik = (g.norm(dim=1) - 1).pow(2).mean()
            return huber(y).mean() + 0.1*eik
        else:
            # primitives: unsigned distance
            if hasattr(patch, "unsigned_distance"):
                d = patch.unsigned_distance(X)  # cylinder
            else:
                d = patch.sdf(X).abs()
            return huber(d).mean()
    raise ValueError("Unknown patch type")

def E_G1(i_patch, j_patch, pi: Tensor, pj: Tensor) -> Tensor:
    ni = i_patch.normal(pi); nj = j_patch.normal(pj)
    return (1.0 - (ni*nj).sum(dim=1).abs()).mean()

def E_G2(i_patch, j_patch, pi: Tensor, pj: Tensor) -> Tensor:
    k1i, k2i = i_patch.principal_curvatures(pi)
    k1j, k2j = j_patch.principal_curvatures(pj)
    ki = torch.sort(torch.stack([k1i.abs(), k2i.abs()], dim=1), dim=1).values
    kj = torch.sort(torch.stack([k1j.abs(), k2j.abs()], dim=1), dim=1).values
    return (ki - kj).pow(2).mean()

def E_coax(i_patch, j_patch, pi: Tensor, pj: Tensor) -> Tensor:
    ui = getattr(i_patch, "characteristic_direction", lambda: None)()
    uj = getattr(j_patch, "characteristic_direction", lambda: None)()
    if ui is None or uj is None:
        return torch.tensor(0.0, device=pi.device)
    ui = ui / (ui.norm()+1e-12); uj = uj / (uj.norm()+1e-12)
    ang = 1.0 - (ui @ uj).abs()
    ci = pi.mean(dim=0); cj = pj.mean(dim=0)
    v = cj - ci; d_ax = (v - (v @ ui)*ui).norm()
    return ang + d_ax

def E_ang(i_patch, j_patch) -> Tensor:
    ui = getattr(i_patch, "characteristic_direction", lambda: None)()
    uj = getattr(j_patch, "characteristic_direction", lambda: None)()
    if ui is None or uj is None:
        return torch.tensor(0.0)
    ui = ui / (ui.norm()+1e-12); uj = uj / (uj.norm()+1e-12)
    return 1.0 - (ui @ uj).abs()

def E_gap(pi: Tensor, pj: Tensor, i_patch=None, j_patch=None, eps: float=1e-3) -> Tensor:
    """
    Gap computed after projecting boundary samples onto current patch surfaces.
    If i_patch/j_patch provided and have `project`, use projected points; else fallback to raw pi/pj.
    Returns squared-clamped mean of distances (soft gap penalty).
    """
    # try to project sample pairs to current patch geometry (so gap depends on params)
    if i_patch is not None and j_patch is not None and hasattr(i_patch, "project") and hasattr(j_patch, "project"):
        try:
            Xi = i_patch.project(pi)
            Yj = j_patch.project(pj)
        except Exception:
            Xi = pi
            Yj = pj
    else:
        Xi = pi
        Yj = pj

    d = (Xi - Yj).norm(dim=1) - eps
    return torch.clamp(d, min=0.0).pow(2).mean()


def E_self(X: Tensor, Y: Tensor, tau: float=1e-3) -> Tensor:
    d = torch.cdist(X, Y)
    return torch.log1p(torch.exp(tau - d)).mean()

def E_reg(patches: Dict[int, object]) -> Tensor:
    reg = 0.0
    for p in patches.values():
        if isinstance(p, (PlanePatch,)):
            reg = reg + (p.n.norm() - 1).pow(2)
        if isinstance(p, (CylinderPatch, ConePatch)):
            reg = reg + (p.a.norm() - 1).pow(2)
        if isinstance(p, CylinderPatch):
            reg = reg + (p.r.abs()*1e-3).pow(2)
    if not isinstance(reg, torch.Tensor):
        reg = torch.tensor(reg, dtype=torch.float32)
    return reg

def total_energy(graph: Graph, patches: Dict[int, object], samples: Dict[int, Tensor], w: Weights) -> Tuple[Tensor, Dict[str,float]]:
    # Data term
    Edata = 0.0
    for pid, patch in patches.items():
        if pid not in samples:
            continue
        Edata = Edata + E_data_patch(patch, samples[pid])
    if not isinstance(Edata, torch.Tensor):
        Edata = torch.tensor(Edata, dtype=torch.float32)

    # Constraint terms
    EG1 = Eang = Ecoax = Egap = EG2 = Eself = torch.tensor(0.0)
    for (i,j), ed in graph.edges.items():
        pi, pj = ed.pairs_i, ed.pairs_j
        EG1 = EG1 + E_G1(patches[i], patches[j], pi, pj)
        EG2 = EG2 + E_G2(patches[i], patches[j], pi, pj)
        Ecoax = Ecoax + E_coax(patches[i], patches[j], pi, pj)
        Eang = Eang + E_ang(patches[i], patches[j])
        Egap = Egap + E_gap(pi, pj, i_patch=patches[i], j_patch=patches[j], eps=1e-3)
        # self-intersections among this pair (sampled)
        X = patches[i].project(pi) if hasattr(patches[i], "project") else pi
        Y = patches[j].project(pj) if hasattr(patches[j], "project") else pj
        Eself = Eself + E_self(X, Y, tau=1e-3)

    Econstr = (w.w_G1*EG1 + w.w_G2*EG2 + w.w_coax*Ecoax + w.w_ang*Eang + w.w_gap*Egap + w.w_self*Eself)

    Ereg = E_reg(patches)

    Etotal = Edata + w.lambda_c*Econstr + w.lambda_r*Ereg

    scalars = dict(Edata=float(Edata.detach()),
                   EG1=float(EG1.detach()), EG2=float(EG2.detach()),
                   Ecoax=float(Ecoax.detach()), Eang=float(Eang.detach()),
                   Egap=float(Egap.detach()), Eself=float(Eself.detach()),
                   Ereg=float(Ereg.detach()), Etotal=float(Etotal.detach()))
    return Etotal, scalars
