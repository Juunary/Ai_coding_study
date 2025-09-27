
from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import numpy as np
from .primitives import PlanePatch, CylinderPatch, SpherePatch, ConePatch
from ..utils import log

Tensor = torch.Tensor

def _to_t(t: np.ndarray) -> Tensor:
    return torch.from_numpy(t.astype(np.float32))

def fit_plane(points: np.ndarray) -> PlanePatch:
    """PCA plane fit: normal = smallest eigenvector; offset by mean."""
    P = points.astype(np.float64)
    mu = P.mean(axis=0)
    C = np.cov((P - mu).T)
    w, v = np.linalg.eigh(C)
    n = v[:, 0].astype(np.float32)
    d = -np.dot(n, mu).astype(np.float32)
    return PlanePatch(torch.nn.Parameter(torch.from_numpy(n)),
                      torch.nn.Parameter(torch.tensor([d], dtype=torch.float32)))

def fit_cylinder(points: np.ndarray) -> CylinderPatch:
    """Axis by PCA first component; radius by mean radial distance; center by mean."""
    P = points.astype(np.float64)
    mu = P.mean(axis=0)
    C = np.cov((P - mu).T)
    w, v = np.linalg.eigh(C)
    a = v[:, -1]  # largest variance direction
    a = a / np.linalg.norm(a)
    # radius: distance to axis
    V = P - mu
    vpar = (V @ a)[:, None] * a[None, :]
    vperp = V - vpar
    r = np.mean(np.linalg.norm(vperp, axis=1))
    return CylinderPatch(torch.nn.Parameter(torch.from_numpy(a.astype(np.float32))),
                         torch.nn.Parameter(torch.from_numpy(mu.astype(np.float32))),
                         torch.nn.Parameter(torch.tensor([r], dtype=torch.float32)))

def fit_sphere(points: np.ndarray) -> SpherePatch:
    """Algebraic sphere fit via least squares."""
    P = points.astype(np.float64)
    A = np.hstack((2*P, np.ones((P.shape[0],1))))
    b = (P**2).sum(axis=1, keepdims=True)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = x[:3, 0]; r = math.sqrt(x[3,0] + (c**2).sum())
    return SpherePatch(torch.nn.Parameter(torch.from_numpy(c.astype(np.float32))),
                       torch.nn.Parameter(torch.tensor([r], dtype=torch.float32)))

def fit_cone(points: np.ndarray) -> ConePatch:
    """Heuristic: axis via PCA; apex by line regression of radius vs axis-projection;
    angle from slope. Works reasonably if the segment is a single conical frustum part."""
    P = points.astype(np.float64)
    mu = P.mean(axis=0)
    C = np.cov((P - mu).T)
    w, v = np.linalg.eigh(C)
    a = v[:, -1]; a = a / np.linalg.norm(a)
    V = P - mu
    z = V @ a  # axial coord
    r = np.linalg.norm(V - np.outer(z, a), axis=1)
    # Linear fit r = s*(z - z0)
    A = np.vstack([z, np.ones_like(z)]).T
    s, b = np.linalg.lstsq(A, r, rcond=None)[0]
    ang = np.arctan(abs(s))
    # approximate apex point at z0 along -a from mu
    z0 = b / (s + 1e-9)
    vtx = mu - z0 * a
    return ConePatch(torch.nn.Parameter(torch.from_numpy(a.astype(np.float32))),
                     torch.nn.Parameter(torch.from_numpy(vtx.astype(np.float32))),
                     torch.nn.Parameter(torch.tensor([ang], dtype=torch.float32)))

def initial_fit(points: np.ndarray, labels: np.ndarray, types: np.ndarray):
    """Return dict patch_id->patch instance based on types mapping.
    types: 0=plane,1=cylinder,2=sphere,3=cone,4=inr
    """
    patches = {}
    K = int(types.shape[0])
    for k in range(K):
        pts = points[labels==k]
        t = int(types[k])
        if t==0:
            patches[k] = fit_plane(pts)
        elif t==1:
            patches[k] = fit_cylinder(pts)
        elif t==2:
            patches[k] = fit_sphere(pts)
        elif t==3:
            patches[k] = fit_cone(pts)
        elif t==4:
            from .inr_sdf import INRSDF, MLP
            model = MLP(d_in=3, d_hidden=64, n_layers=4)
            patches[k] = INRSDF(model)
        else:
            raise ValueError(f"Unknown type {t}")
    log.info(f"Initial patches: {len(patches)}")
    return patches
