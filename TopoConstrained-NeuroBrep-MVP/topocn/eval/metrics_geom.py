
from __future__ import annotations
import torch

def chamfer(P: torch.Tensor, Q: torch.Tensor) -> float:
    dPQ = torch.cdist(P, Q)
    return float(dPQ.min(dim=1).values.mean() + dPQ.min(dim=0).values.mean())

def hausdorff(P: torch.Tensor, Q: torch.Tensor) -> float:
    dPQ = torch.cdist(P, Q)
    return float(torch.max(torch.cat([dPQ.min(dim=1).values, dPQ.min(dim=0).values], dim=0)))
