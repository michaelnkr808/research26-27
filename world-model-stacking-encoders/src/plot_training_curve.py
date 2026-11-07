# Versions used: Python 3.11.14, matplotlib 3.11.2 (also printed at run time).
# Run with /Users/michaelr808/Desktop/research/research26-27/vis-aria-depth/.venv/bin/python
#
# Slide figure: trial (pilot2) DINOv3 run vs two full DINOv3 training runs.
# Plots the logged total training loss against step. NO smoothing; raw logged values.
# The DINOv3 + C-RADIOv4 trial is deliberately excluded (extra loss term, different scale).
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "..")
LOGS = os.path.join(DIR, "experiments", "2026-10-27-proxy-calibration", "results", "logs")
OUT = os.path.join(DIR, "writing", "figures", "2026-11-07-training-curve-trial-vs-full.png")

RUNS = {
    "full1": "train-wave1-dinov3-s1-sh9hj.log",
    "full2": "train-wave1c-dinov3-s0-a40-bj6bf.log",
    "trial": "pilot2-dinov3-kcv7v.log",
}


def read(name):
    pts = []
    with open(os.path.join(LOGS, name)) as f:
        for line in f:
            i = line.find("{")
            if i < 0 or '"step"' not in line:
                continue
            try:
                d = json.loads(line[i:])
            except ValueError:
                continue
            if "step" in d and "loss" in d:
                pts.append((d["step"], d["loss"]))
    return pts


data = {k: read(v) for k, v in RUNS.items()}

print("python", sys.version.split()[0], "matplotlib", matplotlib.__version__)
for k, v in data.items():
    print(f"{k} ({RUNS[k]}): {len(v)} points, steps {v[0][0]}..{v[-1][0]}")
ts, tl = data["trial"][-1]
print(f"trial last logged step {ts}, loss {tl}")
for k in ("full1", "full2"):
    le = [p for p in data[k] if p[0] <= 500][-1]
    print(f"{k}: last logged step <=500 is {le[0]} loss {le[1]}; final step {data[k][-1][0]} loss {data[k][-1][1]}")

BLUE, BLUE2, ORANGE, INK, GREY = "#1f5fa8", "#6fa3d8", "#d9480f", "#1a1a1a", "#555555"
plt.rcParams.update({"font.size": 17, "font.family": "sans-serif"})
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=200)
fig.patch.set_facecolor("white")

for key, col, lab in (("full1", BLUE, "Full training, run 1"), ("full2", BLUE2, "Full training, run 2")):
    x, y = zip(*data[key])
    ax.plot(x, y, color=col, lw=3, marker="o", ms=4, label=lab)
x, y = zip(*data["trial"])
ax.plot(x, y, color=ORANGE, lw=4, label="Trial model", zorder=5)
ax.scatter([ts], [tl], color=ORANGE, s=110, zorder=6, edgecolor="white", linewidth=2)

ax.set_xlim(-300, 22000)
ax.set_xlabel("Training steps", fontsize=19, color=INK)
ax.set_ylabel("Training error (lower is better)", fontsize=19, color=INK)
ax.set_title("The trial models stopped very early", fontsize=26, fontweight="bold", color=INK, loc="left", pad=34)
ax.text(0, 1.03, "Trial: 500 steps. Full runs: about 21,700 steps each.", transform=ax.transAxes,
        fontsize=16, color=GREY, va="bottom")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e5e5e5", lw=1)
ax.set_axisbelow(True)
ax.tick_params(labelsize=16)
ax.set_xticks([0, 5000, 10000, 15000, 20000])
ax.set_xticklabels(["0", "5,000", "10,000", "15,000", "20,000"])

ymax = max(p[1] for k in data for p in data[k])
ax.annotate("Trial model stopped here\n(~500 steps)", xy=(ts, tl), xytext=(4200, ymax * 0.80),
            fontsize=18, fontweight="bold", color=ORANGE, va="center",
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.5))
ax.legend(loc="upper right", fontsize=16, frameon=False)

fig.text(0.01, 0.01, "Raw logged values, no smoothing. Dots are logged points (full runs log every ~433 steps; trial every 10).",
         fontsize=11, color=GREY, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT, dpi=200, facecolor="white")
print("wrote", OUT)
