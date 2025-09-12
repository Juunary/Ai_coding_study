# make_paper_plots.py
# pip install matplotlib pandas numpy

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOG_PATH = r"C:\Users\user\Downloads\evaluation_imp2.log"
OUT_DIR  = r"C:\Users\user\Downloads\paper_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

k2e = {
    "절대 위치 (평균)": "Absolute translation (mean) [mm]",
    "절대 위치 (최대)": "Absolute translation (max) [mm]",
    "상대 위치 (평균)": "Relative ICP (mean) [mm]",
    "상대 위치 (최대)": "Relative ICP (max) [mm]",
    "길이 정확도 (X)": "Length error X [mm]",
    "길이 정확도 (Y)": "Length error Y [mm]",
    "길이 정확도 (Z)": "Length error Z [mm]",
    "기하학 (평균)": "Geometric deviation (Chamfer mean) [mm]",
    "기하학 (최대)": "Geometric deviation (Hausdorff max) [mm]",
    "표면 일치 (평균)": "Surface match (mean) [mm]",
    "표면 일치 (최대)": "Surface match (max) [mm]",
    "특정특징 정확도": "Section feature error [mm]",
}
EXCLUDE = {"부피 오차(%)"}

line_re = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s*\|\s*(?P<metric>[^|]+?)\s*\|\s*(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)
m2_re = re.compile(r"^Mesh2\(test\):\s*(?P<name>.+)$")

data = {}
cur = None
with open(LOG_PATH, "r", encoding="utf-8") as f:
    for raw in f:
        s = raw.strip()
        m2 = m2_re.match(s)
        if m2:
            cur = m2.group("name").strip()
            data.setdefault(cur, {})
            continue
        m = line_re.match(s)
        if m and cur:
            k = m.group("metric").strip()
            if k in EXCLUDE or k not in k2e:
                continue
            v = float(m.group("value"))
            data[cur][k2e[k]] = v

df = pd.DataFrame.from_dict(data, orient="index").sort_index()
df = df.dropna(axis=1, how="all")
df.to_csv(os.path.join(OUT_DIR, "per_file_metrics.csv"), encoding="utf-8-sig")

means  = df.mean(axis=0)
stds   = df.std(axis=0, ddof=1)
meds   = df.median(axis=0)
p95    = df.quantile(0.95, axis=0, interpolation="linear")
stats  = pd.DataFrame({"Mean":means, "Std":stds, "Median":meds, "P95":p95}).sort_index()
means.to_csv(os.path.join(OUT_DIR, "dataset_means_en.csv"))
stats.to_csv(os.path.join(OUT_DIR, "dataset_stats_en.csv"))

# Bar chart (values on bars). Do not set explicit colors or styles.
fig, ax = plt.subplots(figsize=(10,5))
x = np.arange(len(means.index))
bars = ax.bar(x, means.values)
ax.set_xticks(x)
ax.set_xticklabels(list(means.index), rotation=35, ha="right")
ax.set_ylabel("Error [mm]")
ax.set_title(f"Dataset Means (N={len(df)}) — Volume metric excluded")
for rect, val in zip(bars, means.values):
    ax.annotate(f"{val:.4f}", xy=(rect.get_x()+rect.get_width()/2, rect.get_height()),
                xytext=(0,3), textcoords="offset points", ha="center", va="bottom")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "means_bar_en.png"), dpi=300, bbox_inches="tight")

# LaTeX table (Mean ± Std, plus Median)
lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Per-metric reconstruction errors (mm). Volume error is excluded.}")
lines.append(r"\label{tab:recon_metrics}")
lines.append(r"\begin{tabular}{lrrr}")
lines.append(r"\toprule")
lines.append(r"Metric & Mean $\downarrow$ & Std & Median \\")
lines.append(r"\midrule")
for m in means.index:
    lines.append(f"{m} & {means[m]:.4f} & {stds[m]:.4f} & {meds[m]:.4f} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
with open(os.path.join(OUT_DIR, "dataset_table_en.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("Done. Outputs @", OUT_DIR)
