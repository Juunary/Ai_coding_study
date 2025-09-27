
from __future__ import annotations
from typing import Dict, Tuple
import torch

def count_g1_discontinuities(edge_labels: Dict[Tuple[int,int], dict], tau_deg: float=5.0) -> int:
    tau = 1.0 - torch.cos(torch.tensor(tau_deg * 3.14159265/180.0))
    cnt = 0
    for lab in edge_labels.values():
        if lab.get("G1", 0.0) > float(tau):
            cnt += 1
    return cnt

def step_success(step_path: str) -> bool:
    try:
        import os
        return os.path.exists(step_path) and os.path.getsize(step_path) > 0
    except Exception:
        return False
