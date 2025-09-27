
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
Tensor = torch.Tensor

# ----- Base interface -----
class Patch:
    """Abstract surface patch API for optimization.
    All parameters should be torch.nn.Parameter for gradient-based updates.
    """
    kind: str
    def sdf(self, x: Tensor) -> Tensor: ...
    def project(self, x: Tensor) -> Tensor: ...
    def normal(self, x: Tensor) -> Tensor: ...
    def principal_curvatures(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (k1, k2) principal curvatures at x. For primitives, closed-form; for INR, autograd."""
        raise NotImplementedError()
    def characteristic_direction(self) -> Optional[Tensor]:
        """Axis/normal used by coaxial/parallel constraints. (3,) tensor or None."""
        return None

# ----- Plane -----
@dataclass
class PlanePatch(Patch):
    n: torch.nn.Parameter  # (3,) not necessarily unit; normalized internally
    d: torch.nn.Parameter  # (1,)
    kind: str = "plane"

    def _n(self) -> Tensor:
        return self.n / (self.n.norm() + 1e-12)

    def sdf(self, x: Tensor) -> Tensor:
        n = self._n()
        return x @ n + self.d

    def project(self, x: Tensor) -> Tensor:
        n = self._n()
        return x - (x @ n + self.d)[:, None] * n

    def normal(self, x: Tensor) -> Tensor:
        n = self._n()
        return n.expand_as(x)

    def principal_curvatures(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        k1 = torch.zeros(x.shape[0], device=x.device)
        k2 = torch.zeros_like(k1)
        return k1, k2

    def characteristic_direction(self) -> Tensor:
        return self._n()

# ----- Cylinder -----
@dataclass
class CylinderPatch(Patch):
    a: torch.nn.Parameter  # (3,) axis direction (unit-ish)
    c: torch.nn.Parameter  # (3,) point on axis
    r: torch.nn.Parameter  # (1,) radius > 0
    kind: str = "cylinder"

    def _a(self) -> Tensor:
        return self.a / (self.a.norm() + 1e-12)

    def _radial(self, x: Tensor) -> Tensor:
        a = self._a()
        v = x - self.c
        v_par = (v @ a)[:, None] * a
        v_perp = v - v_par
        return v_perp

    def sdf(self, x: Tensor) -> Tensor:
        v_perp = self._radial(x)
        return v_perp.norm(dim=1) - self.r.abs()

    def project(self, x: Tensor) -> Tensor:
        a = self._a(); v = x - self.c
        v_par = (v @ a)[:, None] * a
        v_perp = v - v_par
        return self.c + v_par + v_perp / (v_perp.norm(dim=1, keepdim=True) + 1e-12) * self.r.abs()

    def normal(self, x: Tensor) -> Tensor:
        v_perp = self._radial(x)
        n = v_perp / (v_perp.norm(dim=1, keepdim=True) + 1e-12)
        return n

    def principal_curvatures(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # k1 = 1/R along radial direction, k2 = 0 along axis
        R = self.r.abs() + 1e-12
        k1 = torch.ones(x.shape[0], device=x.device) / R
        k2 = torch.zeros_like(k1)
        return k1, k2

    def characteristic_direction(self) -> Tensor:
        return self._a()

# ----- Sphere -----
@dataclass
class SpherePatch(Patch):
    c: torch.nn.Parameter  # (3,)
    r: torch.nn.Parameter  # (1,)
    kind: str = "sphere"

    def sdf(self, x: Tensor) -> Tensor:
        return (x - self.c).norm(dim=1) - self.r.abs()

    def project(self, x: Tensor) -> Tensor:
        v = x - self.c
        return self.c + v / (v.norm(dim=1, keepdim=True) + 1e-12) * self.r.abs()

    def normal(self, x: Tensor) -> Tensor:
        v = x - self.c
        return v / (v.norm(dim=1, keepdim=True) + 1e-12)

    def principal_curvatures(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        R = self.r.abs() + 1e-12
        k = torch.ones(x.shape[0], device=x.device) / R
        return k, k

    def characteristic_direction(self) -> None:
        return None

# ----- Cone (right circular) -----
@dataclass
class ConePatch(Patch):
    a: torch.nn.Parameter  # (3,) axis direction
    vtx: torch.nn.Parameter  # (3,) apex point
    ang: torch.nn.Parameter  # (1,) half-angle (radians, > 0)
    kind: str = "cone"

    def _a(self) -> Tensor:
        return self.a / (self.a.norm() + 1e-12)

    def sdf(self, x: Tensor) -> Tensor:
        # unsigned distance approx (for small angles accurate near surface)
        a = self._a()
        v = x - self.vtx
        h = (v @ a)[:, None] * a
        r_vec = v - h
        theta = torch.arctan2(r_vec.norm(dim=1), (v @ a).clamp_min(1e-9))
        return (theta - self.ang.abs()).abs()  # crude, sufficient for constraints

    def project(self, x: Tensor) -> Tensor:
        # crude projection by snapping theta to ang
        a = self._a()
        v = x - self.vtx
        z = (v @ a)
        r = (v - z[:, None]*a).norm(dim=1)
        z = z.clamp_min(1e-9)
        target_r = z * torch.tan(self.ang.abs())
        scale = (target_r / (r + 1e-12)).unsqueeze(1)
        rdir = (v - z[:, None]*a) / (r.unsqueeze(1) + 1e-12)
        return self.vtx + z[:, None]*a + scale * rdir * r

    def normal(self, x: Tensor) -> Tensor:
        # gradient of implicit cone (approx)
        a = self._a()
        v = x - self.vtx
        z = (v @ a)
        rvec = v - z[:, None]*a
        r = rvec.norm(dim=1, keepdim=True) + 1e-12
        # implicit f = r - z*tan(ang) = 0
        grad = rvec / r - torch.tan(self.ang.abs()) * a
        return grad / (grad.norm(dim=1, keepdim=True) + 1e-12)

    def principal_curvatures(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Cones are developable: one principal curvature ~ 0; the other depends on position.
        k1 = torch.zeros(x.shape[0], device=x.device)
        # crude finite curvature along circular direction ~ 1/(z * sin(ang)) 
        a = self._a()
        v = x - self.vtx
        z = (v @ a).abs() + 1e-6
        k2 = 1.0 / (z * torch.sin(self.ang.abs()) + 1e-6)
        return k1, k2

    def characteristic_direction(self) -> Tensor:
        return self._a()
