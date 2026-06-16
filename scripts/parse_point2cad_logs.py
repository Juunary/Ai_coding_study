"""Parse Point2CAD evaluation logs and compute summary statistics
matching the NeuroB-Rep Table 3 (tab:consistency_all) format."""

import re
import numpy as np
from pathlib import Path

METRIC_MAP = {
    "절대 위치 (평균)": "abs_pos_mean",
    "절대 위치 (최대)": "abs_pos_max",
    "상대 위치 (평균)": "rel_pos_mean",
    "상대 위치 (최대)": "rel_pos_max",
    "길이 정확도 (X)": "len_width",
    "길이 정확도 (Y)": "len_depth",
    "길이 정확도 (Z)": "len_height",
    "기하학 (평균)": "geom_mean",
    "기하학 (최대)": "geom_max",
    "표면 일치 (평균)": "surf_mean",
    "표면 일치 (최대)": "surf_max",
    "부피 오차(%)": "vol_err_pct",
}

# Maps to NeuroB-Rep Table 3 row names
TABLE3_ROWS = [
    ("Relative Position (Mean Err)", "rel_pos_mean"),
    ("Relative Position (Max Err)", "rel_pos_max"),
    ("Length (Width)", "len_width"),
    ("Length (Depth)", "len_depth"),
    ("Length (Height)", "len_height"),
    ("Geometric Acc (Mean)", "geom_mean"),
    ("Geometric Acc (Max)", "geom_max"),
    ("Surface Coincidence (Mean)", "surf_mean"),
    ("Surface Coincidence (Max)", "surf_max"),
    ("Shape Deviation", "geom_mean"),  # same as geometric mean
]


def parse_log(path: str) -> list:
    """Parse a Point2CAD evaluation log file and return list of per-run dicts."""
    text = Path(path).read_text(encoding="utf-8")
    blocks = text.split("=" * 20)
    runs = []
    for block in blocks:
        block = block.strip()
        if not block or "측정 대상" not in block:
            continue
        d = {}
        for line in block.splitlines():
            for kr, eng in METRIC_MAP.items():
                if kr in line:
                    val = float(line.strip().split("|")[-1].strip())
                    d[eng] = val
        if d:
            runs.append(d)
    return runs


def summarize(runs: list) -> dict:
    """Compute mean, std, median, p95 for each metric."""
    keys = runs[0].keys()
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in runs if k in r])
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "p95": float(np.percentile(vals, 95)),
            "n": len(vals),
        }
    return summary


def print_table3_format(component: str, summary: dict):
    """Print in the same format as NeuroB-Rep Table 3."""
    print(f"\n{'='*70}")
    print(f"  Point2CAD -- {component} (n={list(summary.values())[0]['n']} runs)")
    print(f"{'='*70}")
    print(f"  {'Metric':<35s} {'Mean':>8s} {'Std':>8s} {'Median':>8s} {'P95':>8s}")
    print(f"  {'-'*67}")
    for label, key in TABLE3_ROWS:
        if key in summary:
            s = summary[key]
            print(f"  {label:<35s} {s['mean']:8.5f} {s['std']:8.5f} {s['median']:8.5f} {s['p95']:8.5f}")


def print_latex_rows(component: str, summary: dict):
    """Print LaTeX table rows for copy-paste into access.tex."""
    print(f"\n% Point2CAD -- {component}")
    for label, key in TABLE3_ROWS:
        if key in summary:
            s = summary[key]
            print(f"    {label:<35s} & {s['mean']:.5f} & {s['std']:.5f} & {s['median']:.5f} & {s['p95']:.5f} \\\\")


if __name__ == "__main__":
    base = Path("c:/Users/user/Documents/GitHub/Ai_coding_study/point2cad/point2cad_test")

    logs = {
        "Impeller": base / "evaluation_imp.log",
        "Shaft": base / "evaluation_shaft.log",
        "Casing": base / "evaluation_casing.log",
    }

    all_summaries = {}
    for comp, path in logs.items():
        if path.exists():
            runs = parse_log(str(path))
            summary = summarize(runs)
            all_summaries[comp] = summary
            print_table3_format(comp, summary)
        else:
            print(f"[SKIP] {path} not found")

    print("\n\n" + "=" * 70)
    print("  LaTeX rows for access.tex")
    print("=" * 70)
    for comp, summary in all_summaries.items():
        print_latex_rows(comp, summary)
