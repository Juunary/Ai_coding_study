"""PCA Canonical Transform Ablation (R2-C13).

Runs the NeuroB-Rep solver with and without PCA normalization of the input
point cloud, then compares convergence speed and final metrics.

Usage:
    python scripts/pca_ablation.py
"""

import sys, os, json, time
import numpy as np
import torch

# Allow imports from the MVP package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "TopoConstrained-NeuroBrep-MVP"))

from topocn.optimize.solver import solve, SolveConfig, set_seed, prepare_samples
from topocn.optimize.energy import total_energy, Weights
from topocn.topo.graph import build_adjacency
from topocn.topo.constraints import annotate_edges
from topocn.topo.snap import greedy_snap
from topocn.geometry.fitters import initial_fit
from topocn.eval.metrics_geom import chamfer, hausdorff
from topocn.eval.metrics_cad import count_g1_discontinuities
from topocn.utils import log

# ─── PCA canonical transform ──────────────────────────────────────
def pca_canonical(points: np.ndarray):
    """Center and rotate points so that PCA axes align with xyz.
    Returns (transformed_points, centroid, rotation_matrix)."""
    mu = points.mean(axis=0)
    P = points - mu
    C = np.cov(P.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    # Sort descending by eigenvalue (largest variance → X axis)
    order = np.argsort(eigvals)[::-1]
    R = eigvecs[:, order].T  # rows = new axes
    # Ensure right-handed coordinate system
    if np.linalg.det(R) < 0:
        R[2] *= -1
    P_rot = P @ R.T
    return P_rot.astype(np.float32), mu, R


# ─── Solver with iteration history capture ─────────────────────────
def solve_with_history(points, labels, types, cfg: SolveConfig, tag: str = ""):
    """Run the solver and record energy scalars at each iteration."""
    if isinstance(cfg.weights, dict):
        cfg.weights = Weights(**cfg.weights)
    set_seed(cfg.seed)

    patches = initial_fit(points, labels, types)
    graph = build_adjacency(points, labels, k=12, max_pairs=1024)
    graph = annotate_edges(graph, patches)

    opt_params = []
    for p in patches.values():
        for name, val in p.__dict__.items():
            if isinstance(val, torch.nn.Parameter):
                opt_params.append(val)
        if hasattr(p, "model"):
            opt_params += list(p.model.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=cfg.lr)
    samples = prepare_samples(points, labels, types, patches)

    history = []
    t0 = time.time()
    for it in range(1, cfg.iters + 1):
        optimizer.zero_grad()
        E, scal = total_energy(graph, patches, samples, cfg.weights)
        E.backward()
        optimizer.step()
        if it % cfg.snap_every == 0:
            greedy_snap(graph, patches, w=dict(G1=1.0, G2=0.2, GAP=50.0),
                        max_edges=cfg.max_snap_edges, alpha=0.5)
        if it % cfg.relabel_every == 0:
            graph = annotate_edges(graph, patches)
        history.append({"iter": it, **scal})
    elapsed = time.time() - t0

    # Final metrics
    P = torch.from_numpy(points)
    Pfit = []
    for pid, patch in patches.items():
        Pi = P[labels == pid]
        if Pi.shape[0] == 0:
            continue
        if hasattr(patch, "project"):
            Pfit.append(patch.project(Pi))
        else:
            Pfit.append(Pi)
    Pfit = torch.cat(Pfit, dim=0) if len(Pfit) > 0 else P.clone()

    CD = float(chamfer(P, Pfit))
    HD = float(hausdorff(P, Pfit))

    edge_labels = {(i, j): ed.labels for (i, j), ed in graph.edges.items()}
    g1_disc = count_g1_discontinuities(edge_labels, tau_deg=5.0)

    result = {
        "tag": tag,
        "elapsed_s": round(elapsed, 2),
        "final_Etotal": scal["Etotal"],
        "final_Egap": scal["Egap"],
        "final_EG1": scal["EG1"],
        "Chamfer": CD,
        "Hausdorff": HD,
        "G1_discont": g1_disc,
        "history": history,
    }
    return result


# ─── Main ablation ─────────────────────────────────────────────────
def subsample(points, labels, max_per_patch=2000):
    """Subsample points per patch label to keep runtime manageable on CPU."""
    idx = []
    for k in np.unique(labels):
        mask = np.where(labels == k)[0]
        if len(mask) > max_per_patch:
            rng = np.random.RandomState(42)
            mask = rng.choice(mask, max_per_patch, replace=False)
        idx.append(mask)
    idx = np.sort(np.concatenate(idx))
    return points[idx], labels[idx]


def main():
    base = os.path.join(os.path.dirname(__file__), "..", "TopoConstrained-NeuroBrep-MVP")
    asset_dir = os.path.join(base, "assets")

    pts_raw = np.load(os.path.join(asset_dir, "points.npy")).astype(np.float32)
    labels = np.load(os.path.join(asset_dir, "labels.npy")).astype(np.int64)
    types = np.load(os.path.join(asset_dir, "types.npy")).astype(np.int64)

    # Subsample for tractable CPU runtime (INR autograd is expensive)
    pts_raw, labels = subsample(pts_raw, labels, max_per_patch=2000)
    print(f"Subsampled to {pts_raw.shape[0]} points ({np.unique(labels).size} patches)")

    # Solver config matching impeller.yaml
    cfg = SolveConfig(
        seed=123,
        iters=60,
        lr=0.01,
        snap_every=5,
        relabel_every=5,
        max_snap_edges=20,
        device="cpu",
        weights=Weights(
            w_G1=1.0, w_G2=0.2, w_coax=0.5, w_ang=0.2,
            w_gap=10.0, w_self=0.1, lambda_c=1.0, lambda_r=0.0001
        ),
    )

    num_seeds = 5
    results = {"with_pca": [], "without_pca": []}

    # PCA-normalized points
    pts_pca, mu, R = pca_canonical(pts_raw)
    print(f"Raw   mean: {pts_raw.mean(axis=0)}, std: {pts_raw.std(axis=0)}")
    print(f"PCA   mean: {pts_pca.mean(axis=0)}, std: {pts_pca.std(axis=0)}")
    print()

    for seed in range(num_seeds):
        cfg.seed = seed + 100
        print(f"=== Seed {cfg.seed} ===")

        # WITHOUT PCA (raw coordinates)
        print(f"  [without_pca] running...", end=" ", flush=True)
        r_raw = solve_with_history(pts_raw, labels, types, cfg, tag=f"raw_seed{cfg.seed}")
        print(f"Etotal={r_raw['final_Etotal']:.4f}, CD={r_raw['Chamfer']:.6f}, HD={r_raw['Hausdorff']:.6f}, time={r_raw['elapsed_s']}s")
        results["without_pca"].append(r_raw)

        # WITH PCA (canonical transform)
        print(f"  [with_pca]    running...", end=" ", flush=True)
        r_pca = solve_with_history(pts_pca, labels, types, cfg, tag=f"pca_seed{cfg.seed}")
        print(f"Etotal={r_pca['final_Etotal']:.4f}, CD={r_pca['Chamfer']:.6f}, HD={r_pca['Hausdorff']:.6f}, time={r_pca['elapsed_s']}s")
        results["with_pca"].append(r_pca)

    # ─── Summary statistics ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  PCA Ablation Summary (5 seeds)")
    print("=" * 80)
    header = f"  {'Condition':<20s} {'Etotal':>10s} {'Egap':>10s} {'EG1':>10s} {'Chamfer':>10s} {'Hausdorff':>10s} {'G1_disc':>8s} {'Time(s)':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cond in ["without_pca", "with_pca"]:
        runs = results[cond]
        Et = np.array([r["final_Etotal"] for r in runs])
        Eg = np.array([r["final_Egap"] for r in runs])
        E1 = np.array([r["final_EG1"] for r in runs])
        CD = np.array([r["Chamfer"] for r in runs])
        HD = np.array([r["Hausdorff"] for r in runs])
        G1 = np.array([r["G1_discont"] for r in runs])
        T = np.array([r["elapsed_s"] for r in runs])
        label = "With PCA" if cond == "with_pca" else "Without PCA"
        print(f"  {label:<20s} {Et.mean():10.4f} {Eg.mean():10.4f} {E1.mean():10.4f} "
              f"{CD.mean():10.6f} {HD.mean():10.6f} {G1.mean():8.1f} {T.mean():8.1f}")
        print(f"  {'  (± std)':<20s} {Et.std():10.4f} {Eg.std():10.4f} {E1.std():10.4f} "
              f"{CD.std():10.6f} {HD.std():10.6f} {G1.std():8.1f} {T.std():8.1f}")

    # ─── LaTeX rows for paper ──────────────────────────────────────
    print("\n% LaTeX rows for PCA ablation table")
    for cond in ["without_pca", "with_pca"]:
        runs = results[cond]
        Et = np.mean([r["final_Etotal"] for r in runs])
        Eg = np.mean([r["final_Egap"] for r in runs])
        E1 = np.mean([r["final_EG1"] for r in runs])
        CD = np.mean([r["Chamfer"] for r in runs])
        HD = np.mean([r["Hausdorff"] for r in runs])
        Et_s = np.std([r["final_Etotal"] for r in runs])
        CD_s = np.std([r["Chamfer"] for r in runs])
        label = "With PCA" if cond == "with_pca" else "Without PCA"
        print(f"    {label:<16s} & {Et:.4f} $\\pm$ {Et_s:.4f} & {E1:.4f} "
              f"& {Eg:.4f} & {CD:.6f} $\\pm$ {CD_s:.6f} & {HD:.6f} \\\\")

    # ─── Save full results (without history for size) ──────────────
    out_path = os.path.join(os.path.dirname(__file__), "pca_ablation_results.json")
    save_data = {}
    for cond in ["without_pca", "with_pca"]:
        save_data[cond] = []
        for r in results[cond]:
            entry = {k: v for k, v in r.items() if k != "history"}
            # Save convergence curve (every 5th iter for compactness)
            entry["convergence"] = [h for h in r["history"] if h["iter"] % 5 == 0]
            save_data[cond].append(entry)
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
