
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
import numpy as np
import torch

@dataclass
class EdgeData:
    i: int
    j: int
    pairs_i: torch.Tensor   # [M,3] boundary samples from patch i
    pairs_j: torch.Tensor   # [M,3] paired boundary samples from patch j
    # placeholders for labels/metrics
    labels: Dict[str, float] = field(default_factory=dict)

@dataclass
class Graph:
    V: List[int]
    E: List[Tuple[int,int]]
    edges: Dict[Tuple[int,int], EdgeData]

def build_adjacency(P: np.ndarray, labels: np.ndarray, k: int = 12, max_pairs: int = 2048) -> Graph:
    """Build graph from point labels by kNN co-label transitions.
    For every point p with label a, if any of its k nearest neighbors have label b!=a,
    create edge (a,b) and store (p, q_nn) boundary sample pairs.
    """
    P = P.astype(np.float32)
    N = P.shape[0]
    # brute-force kNN (N can be moderate in MVP). Use torch for speed.
    X = torch.from_numpy(P)
    # chunked distance compute
    K = k
    pairs = {}  # (i,j) -> list of (pi, pj)
    bs = 2048
    for s in range(0, N, bs):
        xe = X[s:s+bs]  # [B,3]
        d2 = torch.cdist(xe, X, p=2)  # [B,N]
        knn_idx = torch.topk(d2, k=K+1, largest=False).indices[:, 1:]  # exclude self
        li = torch.from_numpy(labels[s:s+bs])
        for b in range(xe.shape[0]):
            a = int(li[b])
            neigh = knn_idx[b].tolist()
            for idx in neigh:
                b_label = int(labels[idx])
                if b_label==a: 
                    continue
                key = (min(a,b_label), max(a,b_label))
                pi = xe[b].numpy(); pj = X[idx].numpy()
                if key not in pairs: pairs[key] = []
                pairs[key].append((pi, pj) if key==(a,b_label) else (pj, pi))
    # convert to tensors
    edges = {}
    for (i,j), lst in pairs.items():
        if len(lst)==0: 
            continue
        # reservoir sample max_pairs
        if len(lst) > max_pairs:
            idx = np.random.choice(len(lst), size=max_pairs, replace=False)
            lst = [lst[t] for t in idx]
        Ai = torch.from_numpy(np.stack([a for a,_ in lst]).astype(np.float32))
        Aj = torch.from_numpy(np.stack([b for _,b in lst]).astype(np.float32))
        edges[(i,j)] = EdgeData(i=i, j=j, pairs_i=Ai, pairs_j=Aj)
    V = sorted(list(set(labels.tolist())))
    E = list(edges.keys())
    return Graph(V=V, E=E, edges=edges)
