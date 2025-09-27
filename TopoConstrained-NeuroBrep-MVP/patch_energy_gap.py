from pathlib import Path
p = Path("topocn/optimize/energy.py")
txt = p.read_text(encoding="utf-8")

# Replace the E_gap function with projection-based one
old = "def E_gap(pi: Tensor, pj: Tensor, eps: float=1e-3) -> Tensor:\n    d = (pi - pj).norm(dim=1) - eps\n    return torch.clamp(d, min=0.0).pow(2).mean()\n"
new = """def E_gap(pi: Tensor, pj: Tensor, i_patch=None, j_patch=None, eps: float=1e-3) -> Tensor:
    \"\"\"Gap computed after projecting boundary samples onto current patch surfaces.
    If i_patch/j_patch provided and have `project`, use projected points; else fallback to raw pi/pj.
    \"\"\"
    if i_patch is not None and j_patch is not None and hasattr(i_patch, "project") and hasattr(j_patch, "project"):
        try:
            Xi = i_patch.project(pi)
            Yj = j_patch.project(pj)
        except Exception:
            # fallback if projection fails
            Xi = pi
            Yj = pj
    else:
        Xi = pi
        Yj = pj
    d = (Xi - Yj).norm(dim=1) - eps
    return torch.clamp(d, min=0.0).pow(2).mean()\n"""

if old in txt:
    txt = txt.replace(old, new)
else:
    # try to find function def and replace roughly
    txt = txt.replace("def E_gap(pi: Tensor, pj: Tensor, eps: float=1e-3) -> Tensor:", "def E_gap(pi: Tensor, pj: Tensor, i_patch=None, j_patch=None, eps: float=1e-3) -> Tensor:")
    # naive append fallback if not present; make cautious
    if "projected points onto current patch surfaces" not in txt:
        txt = txt.replace("def E_self(X: Tensor, Y: Tensor, tau: float=1e-3) -> Tensor:", new + "\n\ndef E_self(X: Tensor, Y: Tensor, tau: float=1e-3) -> Tensor:")

p.write_text(txt, encoding="utf-8")
print("Patched", p.resolve())
