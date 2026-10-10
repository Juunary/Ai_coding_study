"""Dictionary-based surface type selection (draft refactor).

Re-implements the selection rule of `process_one_surface` in
`fitting_one_surface.py` (the argmin / basic-primitive preference block)
keyed by type name instead of hard-coded integer indices, so that a new
primitive type (e.g. torus) only needs one more entry in `errors`.

Optionally takes a candidate set (e.g. from a retrieval step); types that are
not candidates are masked with inf before the usual rule is applied.
"""
import numpy as np

BASIC_ERROR_THRES = 0.008
BASIC_ERROR_INCREMENT_THRES = 0.001
FREEFORM = "inr"


def select_surface_type(errors, candidates=None):
    """errors: {type_name: fit error}; must contain FREEFORM ("inr").
    candidates: optional iterable of type names allowed to win (inr is always allowed).
    Returns the selected type name.
    """
    errs = {k: (np.inf if v is None or not np.isfinite(v) else float(v)) for k, v in errors.items()}
    if candidates is not None:
        allowed = set(candidates) | {FREEFORM}
        errs = {k: (v if k in allowed else np.inf) for k, v in errs.items()}

    basic = {k: v for k, v in errs.items() if k != FREEFORM}
    best_basic = min(basic, key=basic.get)
    best_all = min(errs, key=errs.get)

    if best_all != FREEFORM:
        chosen = best_all
    elif basic[best_basic] < BASIC_ERROR_THRES:
        chosen = best_basic
    elif basic[best_basic] < errs[FREEFORM] + BASIC_ERROR_INCREMENT_THRES:
        chosen = best_basic
    else:
        chosen = FREEFORM

    # original special case: a cone that is indistinguishable from a plane is a plane
    if chosen == "cone" and "plane" in errs and abs(errs["plane"] - errs["cone"]) <= 1e-5:
        chosen = "plane"
    return chosen
