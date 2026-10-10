"""XYZ -> shape elements -> fixed-length descriptor.

Raw coordinates depend on position, orientation and size, so they are first
rewritten as local differential-geometry elements of the surface:

  per point : normal, principal curvatures k1/k2, curvedness, |shape index|,
              Gaussian-curvature sign, surface variation, normal alignment
              with the patch's normal-tensor axes
  per patch : normal-tensor eigenvalues, point-spread eigenvalues,
              curvature variation

All elements are invariant to translation, rotation and (after normalization)
scale. `shape_descriptor` summarizes their distribution into one vector, which
is the input of the embedding model in `shape_encoder.py`.
"""
import numpy as np

K_NEIGHBORS = 32
MAX_POINTS = 1024
IRLS_ITERS = 3


def normalize_patch(points, max_points=MAX_POINTS, seed=0):
    p = np.asarray(points, dtype=np.float64)[:, :3]
    if len(p) > max_points:
        p = p[np.random.default_rng(seed).choice(len(p), max_points, replace=False)]
    p = p - p.mean(axis=0)
    scale = np.sqrt((p ** 2).sum(axis=1).mean())
    return p / max(scale, 1e-12)


def _knn(p, k):
    d = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
    k = min(k, len(p) - 1)
    return np.argpartition(d, k, axis=1)[:, : k + 1]


def point_elements(points, k=K_NEIGHBORS):
    """Returns a dict of per-point / per-patch elements for a normalized patch."""
    p = normalize_patch(points)
    nb = p[_knn(p, k)]                                    # (N, k+1, 3)

    q = nb - nb.mean(axis=1, keepdims=True)
    w, v = np.linalg.eigh(np.einsum("nki,nkj->nij", q, q) / q.shape[1])
    normal, t2, t1 = v[:, :, 0], v[:, :, 1], v[:, :, 2]
    surface_variation = w[:, 0] / np.maximum(w.sum(axis=1), 1e-12)

    # local quadric h = a x^2 + b xy + c y^2 + d x + e y + f in the tangent frame
    r = nb - p[:, None, :]
    x = np.einsum("nki,ni->nk", r, t1)
    y = np.einsum("nki,ni->nk", r, t2)
    h = np.einsum("nki,ni->nk", r, normal)
    A = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], axis=-1)
    # robust (IRLS, Cauchy weights) so that a few off-surface points in the
    # neighbourhood, e.g. boundary points of an adjacent face, do not bend the fit
    wts = np.ones_like(h)
    for _ in range(IRLS_ITERS + 1):
        AtA = np.einsum("nki,nk,nkj->nij", A, wts, A) + 1e-9 * np.eye(6)
        coef = np.linalg.solve(AtA, np.einsum("nki,nk,nk->ni", A, wts, h)[..., None])[..., 0]
        res = h - np.einsum("nki,ni->nk", A, coef)
        scale = 1.4826 * np.median(np.abs(res), axis=1, keepdims=True) + 1e-4
        wts = 1 / (1 + (res / (2.5 * scale)) ** 2)
    a, b, c, d, e = coef[:, 0], coef[:, 1], coef[:, 2], coef[:, 3], coef[:, 4]

    # fundamental forms of the graph surface at the point
    E, F, G = 1 + d * d, d * e, 1 + e * e
    s = np.sqrt(1 + d * d + e * e)
    L, M, N = 2 * a / s, b / s, 2 * c / s
    detI = E * G - F * F
    K = (L * N - M * M) / detI
    H = (E * N - 2 * F * M + G * L) / (2 * detI)
    disc = np.sqrt(np.maximum(H * H - K, 0))
    k1, k2 = H + disc, H - disc

    curvedness = np.sqrt((k1 * k1 + k2 * k2) / 2)
    # |shape index|: 1 = sphere-like, 0.5 = cylinder-like, 0 = saddle (normal-sign invariant)
    shape_index = np.abs(np.arctan2(k1 + k2, k1 - k2)) * 2 / np.pi
    gauss_sign = K / (curvedness ** 2 + 1e-9)            # +1 sphere, 0 cylinder, -1 saddle

    # normal tensor of the whole patch (sign invariant); its axes act as the patch's own frame
    T = normal.T @ normal / len(normal)
    tw, tv = np.linalg.eigh(T)
    alignment = np.abs(normal @ tv)                       # columns: smallest .. largest axis

    return {
        "points": p,
        "normal": normal,
        "k1": k1,
        "k2": k2,
        "curvedness": curvedness,
        "shape_index": shape_index,
        "gauss_sign": np.clip(gauss_sign, -1, 1),
        "surface_variation": surface_variation,
        "alignment": alignment,
        "normal_tensor_eig": tw,
        "spread_eig": np.linalg.eigvalsh(p.T @ p / len(p)),
    }


def _hist(x, lo, hi, bins):
    h, _ = np.histogram(np.clip(x, lo, hi), bins=bins, range=(lo, hi))
    return h / max(h.sum(), 1)


def shape_descriptor(points, k=K_NEIGHBORS):
    el = point_elements(points, k)
    logc = np.log10(el["curvedness"] + 1e-3)
    joint, _, _ = np.histogram2d(
        np.clip(logc, -2, 1), el["gauss_sign"], bins=6, range=((-2, 1), (-1, 1))
    )
    joint = joint.ravel() / max(joint.sum(), 1)
    c = el["curvedness"]
    stats = np.array([
        logc.mean(),
        logc.std(),
        c.std() / (c.mean() + 0.1),                       # offset keeps flat patches bounded
        np.percentile(logc, 10),
        np.percentile(logc, 90),
    ])
    return np.concatenate([
        joint,                                             # curvedness x Gaussian sign
        _hist(el["shape_index"], 0, 1, 8),
        _hist(el["gauss_sign"], -1, 1, 8),
        _hist(el["alignment"][:, 0], 0, 1, 6),
        _hist(el["alignment"][:, 1], 0, 1, 6),
        _hist(el["alignment"][:, 2], 0, 1, 6),
        _hist(np.log10(el["surface_variation"] + 1e-6), -6, np.log10(1 / 3), 6),
        el["normal_tensor_eig"],
        el["spread_eig"] / el["spread_eig"].sum(),
        stats,
    ]).astype(np.float64)
