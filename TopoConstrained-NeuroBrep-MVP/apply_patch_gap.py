# apply_patch_gap.py
from pathlib import Path
p = Path("topocn/optimize/energy.py")
txt = p.read_text(encoding="utf-8")

# replace simple E_gap definition if present
if "def E_gap(pi: Tensor, pj: Tensor, eps: float=1e-3) -> Tensor:" in txt:
    txt = txt.replace(
        "def E_gap(pi: Tensor, pj: Tensor, eps: float=1e-3) -> Tensor:\n    d = (pi - pj).norm(dim=1) - eps\n    return torch.clamp(d, min=0.0).pow(2).mean()\n",
        """def E_gap(pi: Tensor, pj: Tensor, i_patch=None, j_patch=None, eps: float=1e-3) -> Tensor:
    \"\"\"Gap computed after projecting boundary samples onto current patch surfaces.
    If i_patch/j_patch provided and have `project`, use projected points; else fallback to raw pi/pj.\"\"\"
    if i_patch is not None and j_patch is not None and hasattr(i_patch, \"project\") and hasattr(j_patch, \"project\"):
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
    return torch.clamp(d, min=0.0).pow(2).mean()\n"""
    )

# replace call inside total_energy
txt = txt.replace("Egap = Egap + E_gap(pi, pj, eps=1e-3)", "Egap = Egap + E_gap(pi, pj, i_patch=patches[i], j_patch=patches[j], eps=1e-3)")

p.write_text(txt, encoding="utf-8")
print("Patched energy.py")
