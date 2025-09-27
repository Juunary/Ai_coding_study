
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
Tensor = torch.Tensor

class MLP(nn.Module):
    def __init__(self, d_in=3, d_hidden=64, n_layers=4):
        super().__init__()
        layers = []
        for i in range(n_layers):
            inp = d_in if i==0 else d_hidden
            layers += [nn.Linear(inp, d_hidden), nn.ReLU(inplace=True)]
        layers += [nn.Linear(d_hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

@dataclass
class INRSDF:
    """Implicit neural representation of SDF for a single patch."""
    model: MLP
    kind: str = "inr"

    def sdf(self, x: Tensor) -> Tensor:
        return self.model(x).squeeze(-1)

    def normal(self, x: Tensor) -> Tensor:
        x = x.requires_grad_(True)
        y = self.sdf(x)
        grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        n = grad / (grad.norm(dim=1, keepdim=True) + 1e-12)
        return n

    def curvature_shape_op(self, x: Tensor) -> Tensor:
        """Return shape operator S at x for G2 via Hessian (projected)."""
        x = x.requires_grad_(True)
        y = self.sdf(x)
        J = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]  # grad
        H_rows = []
        for k in range(3):
            gk = J[:, k]
            hk = torch.autograd.grad(gk, x, grad_outputs=torch.ones_like(gk), create_graph=True)[0]
            H_rows.append(hk.unsqueeze(1))
        H = torch.cat(H_rows, dim=1)  # [B,3,3]
        n = J / (J.norm(dim=1, keepdim=True) + 1e-12)
        I = torch.eye(3, device=x.device).unsqueeze(0).expand(x.shape[0],3,3)
        P = I - n.unsqueeze(2) @ n.unsqueeze(1)
        # shape operator S = - P H P / ||grad||
        gn = J.norm(dim=1, keepdim=True) + 1e-12
        S = - (P @ H @ P) / gn.unsqueeze(2)
        return S

    def principal_curvatures(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        S = self.curvature_shape_op(x)  # [B,3,3], rank-2 in tangent
        # eigenvalues of S in tangent plane: take top-2 by abs
        evals = torch.linalg.eigvals(S).real  # [B,3]
        # discard near-zero along normal by magnitude sort
        abs_e = evals.abs()
        idx = torch.argsort(abs_e, dim=1, descending=True)
        k1 = evals.gather(1, idx[:, :1]).squeeze(1)
        k2 = evals.gather(1, idx[:, 1:2]).squeeze(1)
        return k1, k2
