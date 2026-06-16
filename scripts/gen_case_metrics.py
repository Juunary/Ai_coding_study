"""
Regenerate case_metrics_mean.png using Table 7 values from access.tex.
Two cases:
  Case 1 (blue)   = complex misassignment region  → Casing component
  Case 2 (orange) = representative mesh example   → Shaft component
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- metric labels (match original figure x-axis) ---
labels = [
    "Rel. pos.\n(mean)",
    "Rel. pos.\n(max)",
    "Length\n(W)",
    "Length\n(D)",
    "Length\n(H)",
    "Geom. acc.\n(mean)",
    "Geom. acc.\n(max)",
    "Surf. coinc.\n(mean)",
    "Surf. coinc.\n(max)",
    "Shape\ndev.",
]

# --- Casing (complex misassignment case) — Table 7 NeuroB-Rep ---
casing_mean = np.array([0.00692, 0.02546, 0.01080, 0.00983, 0.02283,
                         0.00992, 0.03498, 0.00992, 0.03335, 0.00992])
casing_std  = np.array([0.00006, 0.00137, 0.00740, 0.00730, 0.01412,
                         0.00009, 0.00234, 0.00009, 0.00204, 0.00009])

# --- Shaft (representative mesh example) — Table 7 NeuroB-Rep ---
shaft_mean  = np.array([0.00304, 0.01204, 0.00311, 0.00630, 0.00947,
                         0.00495, 0.01652, 0.00496, 0.01592, 0.00495])
shaft_std   = np.array([0.00002, 0.00263, 0.00158, 0.00567, 0.00657,
                         0.00065, 0.00358, 0.00068, 0.00365, 0.00065])

# --- layout ---
n = len(labels)
x = np.arange(n)
w = 0.35  # bar width

fig, ax = plt.subplots(figsize=(10, 3.8))

bars1 = ax.bar(x - w/2, casing_mean, w,
               yerr=casing_std, capsize=3,
               color="#4472C4", label="Complex misassignment region (Casing)",
               error_kw=dict(ecolor="black", elinewidth=0.8))

bars2 = ax.bar(x + w/2, shaft_mean, w,
               yerr=shaft_std, capsize=3,
               color="#ED7D31", label="Representative mesh example (Shaft)",
               error_kw=dict(ecolor="black", elinewidth=0.8))

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("Error (mm)", fontsize=9)
ax.set_ylim(bottom=0)
ax.yaxis.grid(True, linewidth=0.4, alpha=0.7)
ax.set_axisbelow(True)
ax.legend(fontsize=7.5, loc="upper right")

# light horizontal reference lines for acceptance thresholds (optional context)
for threshold, ls in [(0.01, "--"), (0.05, ":")]:
    ax.axhline(threshold, color="gray", linewidth=0.6, linestyle=ls, alpha=0.5)

fig.tight_layout()
out = r"c:\Users\user\Documents\GitHub\Ai_coding_study\case_metrics_mean.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
