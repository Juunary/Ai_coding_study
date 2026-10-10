"""Synthetic surface patches for building / training the shape vector DB.

Each generator samples a partial patch (random size, angular extent, pose,
point count and noise) so that the stored examples resemble segmented clusters.
"""
import numpy as np

TYPES = ("plane", "sphere", "cylinder", "cone", "torus")


def _rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.diag(r))


def _plane(rng, n):
    if rng.uniform() < 0.5:                                # rectangle
        a, b = rng.uniform(0.5, 2.0, 2)
        return np.stack([rng.uniform(-a, a, n), rng.uniform(-b, b, n), np.zeros(n)], 1)
    r_out = rng.uniform(0.5, 2.0)                          # annulus (flange / end face)
    r_in = r_out * rng.uniform(0.0, 0.9)
    r = np.sqrt(rng.uniform(r_in ** 2, r_out ** 2, n))
    ph = rng.uniform(0, np.deg2rad(rng.uniform(90, 360)), n)
    return np.stack([r * np.cos(ph), r * np.sin(ph), np.zeros(n)], 1)


def _sphere(rng, n):
    R = rng.uniform(0.5, 3.0)
    tmax = np.deg2rad(rng.uniform(40, 180))
    pmax = np.deg2rad(rng.uniform(120, 360))
    ct = rng.uniform(np.cos(tmax), 1, n)
    st = np.sqrt(1 - ct ** 2)
    ph = rng.uniform(0, pmax, n)
    return R * np.stack([st * np.cos(ph), st * np.sin(ph), ct], 1)


def _cylinder(rng, n):
    R = rng.uniform(0.3, 2.0)
    H = rng.uniform(0.3, 3.0)
    ph = rng.uniform(0, np.deg2rad(rng.uniform(90, 360)), n)
    return np.stack([R * np.cos(ph), R * np.sin(ph), rng.uniform(0, H, n)], 1)


def _cone(rng, n):
    al = np.deg2rad(rng.uniform(15, 65))
    t0 = rng.uniform(0.2, 1.0)
    t1 = t0 + rng.uniform(0.5, 3.0)
    t = np.sqrt(rng.uniform(t0 ** 2, t1 ** 2, n))        # area density grows with t
    ph = rng.uniform(0, np.deg2rad(rng.uniform(120, 360)), n)
    return t[:, None] * np.stack(
        [np.sin(al) * np.cos(ph), np.sin(al) * np.sin(ph), np.full(n, np.cos(al))], 1
    )


def _torus(rng, n):
    R = rng.uniform(1.0, 3.0)
    r = R * rng.uniform(0.2, 0.6)
    th0 = rng.uniform(0, 2 * np.pi)
    text = np.deg2rad(rng.uniform(120, 360))
    pmax = np.deg2rad(rng.uniform(60, 360))
    out, total = [], 0
    while total < n:
        th = rng.uniform(th0, th0 + text, 2 * n)
        th = th[rng.uniform(0, R + r, 2 * n) < R + r * np.cos(th)]   # area density
        ph = rng.uniform(0, pmax, len(th))
        out.append(np.stack(
            [(R + r * np.cos(th)) * np.cos(ph), (R + r * np.cos(th)) * np.sin(ph), r * np.sin(th)], 1
        ))
        total += len(th)
    return np.concatenate(out)[:n]


_GEN = {"plane": _plane, "sphere": _sphere, "cylinder": _cylinder, "cone": _cone, "torus": _torus}


def _bleed_edge(p, rng):
    """Bend the points along one edge of the patch out of the surface, like the
    few points of an adjacent face that segmentation assigns to this cluster."""
    c = p - p.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    ang = rng.uniform(0, 2 * np.pi)
    proj = c @ (np.cos(ang) * vt[0] + np.sin(ang) * vt[1])
    q = np.quantile(proj, 1 - rng.uniform(0.02, 0.10))
    edge = proj > q
    out = p.copy()
    out[edge] += np.outer((proj[edge] - q) * np.tan(np.deg2rad(rng.uniform(45, 90 - 1e-3))), vt[2])
    return out


def sample_patch(shape_type, rng, n_range=(150, 1000), noise_range=(0.0, 0.02), clean_prob=0.3,
                 bleed_prob=0.3):
    """clean_prob: share of noise-free patches (points sampled from CAD meshes have no noise).
    bleed_prob: share of patches with a bent edge (segmentation boundary contamination)."""
    n = int(rng.integers(*n_range))
    p = _GEN[shape_type](rng, n)
    if rng.uniform() < bleed_prob:
        p = _bleed_edge(p, rng)
    p = p @ _rotation(rng).T + rng.normal(0, 3, 3)
    scale = np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean())
    noise = 0.0 if rng.uniform() < clean_prob else rng.uniform(*noise_range)
    return p + rng.normal(0, noise * scale, p.shape)


def make_dataset(types, n_per_type, seed=0, **kw):
    rng = np.random.default_rng(seed)
    patches, labels = [], []
    for t in types:
        for _ in range(n_per_type):
            patches.append(sample_patch(t, rng, **kw))
            labels.append(t)
    return patches, np.array(labels)
