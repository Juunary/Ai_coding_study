
from __future__ import annotations
from typing import Dict, Tuple
import torch

def gap_violations(pi: torch.Tensor, pj: torch.Tensor, eps: float=1e-3) -> int:
    d = (pi - pj).norm(dim=1)
    return int((d > eps).sum().item())

def self_intersection_soft(X: torch.Tensor, Y: torch.Tensor, tau: float=1e-3) -> float:
    # count near-penetrations using pairwise distances in a small sample
    d = torch.cdist(X, Y)
    return float((d < tau).float().mean().item())
