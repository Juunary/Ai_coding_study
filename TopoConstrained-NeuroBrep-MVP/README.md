
# TopoConstrained‑NeuroBrep (MVP)

Minimal, research‑grade prototype of constraint/topology‑aware CAD reconstruction from point clouds.
Dataset for this MVP is limited to an in‑house **impeller** scan (see `dataset/` for expected layout).

> Python 3.10+, PyTorch (1.12+), NumPy. Optional: `pythonocc-core` for STEP/IGES export and CAD validity checks.

## Layout
```
topocn/
  geometry/{primitives.py,inr_sdf.py,fitters.py}
  topo/{graph.py,constraints.py,snap.py,validators.py}
  optimize/{energy.py,solver.py}
  export/{brep.py,step_writer.py}
  eval/{metrics_geom.py,metrics_cad.py,bench_runner.py}
configs/impeller.yaml
bin/run_impeller.py
tests/smoke_minimal.py
```
## Quick start (synthetic impeller demo)

```bash
python -m pip install torch numpy pyyaml
python -m pip install pythonocc-core  # optional, for STEP export/validation

# Run minimal synthetic demo (no OCC required)
python tests/smoke_minimal.py

# Or run with a real impeller dataset directory
python bin/run_impeller.py --config configs/impeller.yaml --data_dir /path/to/impeller_dir
```
### Expected impeller data directory
- `points.npy`: float32 [N,3]
- `labels.npy`: int64 [N] (0..K-1)
- `types.npy`:  int64 [K] enumerating {0:plane,1:cylinder,2:sphere,3:cone,4:inr}
- (optional) `scales.json`: {"unit":"mm"}

## Notes
- The MVP prioritizes **clear interfaces and differentiable energy terms** over maximal performance.
- STEP export routes (`export/`) use OpenCascade; if not installed, code falls back to JSON+OBJ with warnings.
- All stochastic components accept `seed` from config for reproducibility.

