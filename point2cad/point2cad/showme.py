import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.unicode_minus'] = False

baseline = {
    "Absolute Pos. (Mean)": 0.044462,
    "Absolute Pos. (Max)": 0.085654,
    "Relative Pos. (Mean)": 0.021235,
    "Relative Pos. (Max)": 0.454875,
    "Length Acc. (X)": 0.577272,
    "Length Acc. (Y)": 0.319519,
    "Length Acc. (Z)": 0.215301,
    "Geometry (Mean)": 0.022715,
    "Geometry (Max)": 0.468135,
    "Surface Match (Mean)": 0.017920,
    "Surface Match (Max)": 0.099473,
}

pipeline = {
    "Absolute Pos. (Mean)": 0.000437,
    "Absolute Pos. (Max)": 0.000771,
    "Relative Pos. (Mean)": 0.007109,
    "Relative Pos. (Max)": 0.028408,
    "Length Acc. (X)": 0.003239,
    "Length Acc. (Y)": 0.000326,
    "Length Acc. (Z)": 0.014288,
    "Geometry (Mean)": 0.009959,
    "Geometry (Max)": 0.035033,
    "Surface Match (Mean)": 0.009916,
    "Surface Match (Max)": 0.033649,
}

labels = list(baseline.keys())
baseline_vals = [baseline[k] for k in labels]
pipeline_vals = [pipeline[k] for k in labels]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12,6))
rects1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline')
rects2 = ax.bar(x + width/2, pipeline_vals, width, label='New Pipeline')

# Y축, 제목, 라벨
ax.set_ylabel('Error Value (mm)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend()

# ───────────── 막대 위에 값 표시 ─────────────
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2,  # x 위치 (막대 가운데)
            height,                               # y 위치 (막대 끝 위)
            f'{height:.3f}',                      # 표시 형식 (소수 3자리)
            ha='center', va='bottom', fontsize=8, rotation=90
        )

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()
