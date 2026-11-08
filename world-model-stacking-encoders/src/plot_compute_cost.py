# Versions used: Python 3.11.14, matplotlib 3.11.2 (also printed at run time).
# Run with /Users/michaelr808/Desktop/research/research26-27/vis-aria-depth/.venv/bin/python
#
# Slide figures for "Main Bottlenecks: Compute".
#  1) 2026-11-07-compute-cost.png: GPU time to train one model vs ESTIMATED GPU time to test one model.
#  2) 2026-11-07-gpu-use-during-testing.png: whole-run average GPU use during testing (separate PNG so each
#     stays readable at half-slide width).
#
# Training time definition: the "[proxy NNNNs] wrote /mnt/proxy/runs/..." line, i.e. seconds on the
# training clock (excludes pod start / image pull / queueing). Mean of the two full runs.
# Testing estimate: 8 chunks x mean(finished-chunk time). Chunk time = the N in "in Ns" on the
# "finished:" line. ESTIMATE only: not every chunk has finished, and chunks on fast GPUs run longer
# because the simulator is CPU-bound.
import glob
import os
import re
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "..")
LOGS = os.path.join(DIR, "experiments", "2026-10-27-proxy-calibration", "results", "logs")
FIG = os.path.join(DIR, "writing", "figures")
OUT1 = os.path.join(FIG, "2026-11-07-compute-cost.png")
OUT2 = os.path.join(FIG, "2026-11-07-gpu-use-during-testing.png")
TRAIN_LOGS = ["train-wave1-dinov3-s1-sh9hj.log", "train-wave1c-dinov3-s0-a40-bj6bf.log"]
N_CHUNKS = 8
MIN_GPU_USE = 40

print("python", sys.version.split()[0], "matplotlib", matplotlib.__version__)

# ---- training
train_s = []
for name in TRAIN_LOGS:
    t = None
    for line in open(os.path.join(LOGS, name)):
        m = re.search(r"\[proxy\s+(\d+)s\] wrote /mnt/proxy/runs/", line)
        if m:
            t = int(m.group(1))
    train_s.append(t)
    print(f"train {name}: {t} s = {t/3600:.2f} h (training clock)")
train_h = statistics.mean(train_s) / 3600
print(f"mean training time: {train_h:.2f} h")

# ---- testing chunks
chunks = []
for p in sorted(glob.glob(os.path.join(LOGS, "eval-dinov3-*.log"))):
    name = os.path.basename(p)
    hit = None
    for line in open(p):
        m = re.search(r"finished: (\d+) records, (\d+) errors in (\d+)s \((\d+) episodes/h\)", line)
        if m:
            hit = tuple(int(x) for x in m.groups())
    if hit is None:
        print(f"EXCLUDED {name}: no 'finished:' line (unfinished or deleted pod)")
        continue
    chunks.append((name,) + hit)
    note = ("  [resumed 135 episodes done earlier on another GPU; time slightly understates a full chunk, kept]"
            if "shards67-0-pn5qf" in name else "")
    print(f"chunk {name}: records {hit[0]}, errors {hit[1]}, {hit[2]} s, {hit[3]} episodes/h{note}")
mean_chunk = statistics.mean(c[3] for c in chunks)
test_h = N_CHUNKS * mean_chunk / 3600
print(f"{len(chunks)} finished chunks; mean chunk time {mean_chunk:.0f} s = {mean_chunk/3600:.2f} h")
print(f"ESTIMATE: test one model = {N_CHUNKS} x mean chunk = {test_h:.1f} GPU-hours "
      f"({test_h/train_h:.1f}x the training time)")

# ---- whole-run GPU use per finished L4 chunk ("mean X% over N samples" line in each eval log)
l4 = []
for c in chunks:
    val = None
    for line in open(os.path.join(LOGS, c[0])):
        m = re.search(r"mean ([\d.]+)% over (\d+) samples", line)
        if m:
            val = (float(m.group(1)), int(m.group(2)))
    print(f"gpu use {c[0]}: whole-run mean {val[0]}% over {val[1]} samples")
    l4.append(val[0])
# Fast-GPU values are NOT in raw files; copied from the experiment LOG.md
# (world-model-stacking-encoders/experiments/2026-10-27-proxy-calibration/LOG.md):
A40_USE = 19.95  # LOG 2026-11-07 (d): cluster monitor 10-min average ~30 min into the A40 run
L40_USE = 22.8   # LOG 2026-11-07 (k): cluster monitor (DCGM) mean over the ~4 h L40 run
print(f"A40 (first 30 min): {A40_USE}% (hardcoded from LOG 2026-11-07 (d))")
print(f"L40 (~4-hour average): {L40_USE}% (hardcoded from LOG 2026-11-07 (k))")

BLUE, ORANGE, INK, GREY = "#1f5fa8", "#d9480f", "#1a1a1a", "#555555"
plt.rcParams.update({"font.size": 17, "font.family": "sans-serif"})

# ---- figure 1
fig, ax = plt.subplots(figsize=(8.5, 7.2), dpi=200)
ax.bar([0, 1], [train_h, test_h], width=0.6, color=[BLUE, ORANGE])
ax.set_xticks([0, 1])
ax.set_xticklabels(["Train one model\n(our small copy)", "Test one model on\nthe full benchmark"], fontsize=17, color=INK)
ax.set_ylabel("GPU-hours (one GPU running for one hour)", fontsize=16, color=INK)
ax.set_ylim(0, test_h * 1.18)
ax.text(0, train_h + test_h * 0.02, f"~{train_h:.0f} GPU-hours", ha="center", va="bottom",
        fontsize=21, fontweight="bold", color=BLUE)
ax.text(1, test_h + test_h * 0.02, f"~{test_h:.0f} GPU-hours\n(estimate)", ha="center",
        va="bottom", fontsize=21, fontweight="bold", color=ORANGE)
ax.set_title("Testing costs far more than training", fontsize=24, fontweight="bold", color=INK, loc="left", pad=14)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e5e5e5", lw=1)
ax.set_axisbelow(True)
ax.tick_params(axis="y", labelsize=15)
fig.text(0.01, 0.01,
         "Estimate: testing is split into 8 chunks, each run on its own GPU.\n"
         f"8 chunks x average finished chunk ({len(chunks)} finished, about {mean_chunk / 3600:.1f} h each).\n"
         "Training ran on an RTX 3090 and an A40; testing on L4s.\n"
         "Data from our own logs, 2026-11-07.",
         fontsize=14, color=GREY, ha="left", va="bottom", linespacing=1.3)
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.savefig(OUT1, dpi=200, facecolor="white")
plt.close(fig)
print("wrote", OUT1)

# ---- figure 2
fig, ax = plt.subplots(figsize=(10, 7.2), dpi=200)
vals = l4 + [A40_USE, L40_USE]
n = len(vals)
cols = [BLUE] * len(l4) + [ORANGE, ORANGE]
ax.bar(range(n), vals, width=0.7, color=cols)
for i, v in enumerate(vals):
    ax.text(i, v - 2, f"{v:.0f}%", ha="center", va="top", fontsize=15, fontweight="bold", color="white")
ax.axhline(MIN_GPU_USE, color=INK, lw=2.5, ls="--", zorder=1)
ax.annotate("Cluster's minimum average GPU use (40%)", xy=(3.5, MIN_GPU_USE), xytext=(-0.4, 78),
            ha="left", va="bottom", fontsize=15, fontweight="bold", color=INK,
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
ax.annotate("Too idle: stopped.\nThe L40 run triggered the\ncluster's idle-GPU warning email",
            xy=(7.5, 26), xytext=(6.4, 58), ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=ORANGE,
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2))
ax.set_ylim(0, 100)
ax.set_xlim(-0.6, 8.6)
ax.set_xlim(-0.6, n - 0.4)
ax.set_xticks(range(n))
ax.set_xticklabels([f"L4 GPU {i + 1}" for i in range(len(l4))] + ["A40 (first 30 min)", "L40 (~4-hour average)"],
                   fontsize=13, color=INK, rotation=0)
for lab in ax.get_xticklabels():
    lab.set_rotation(45)
    lab.set_ha("right")
ax.set_ylabel("GPU busy, average (%)", fontsize=17, color=INK)
ax.set_title("Testing uses about half a GPU,\nand much less on fast GPUs", fontsize=22, fontweight="bold",
             color=INK, loc="left", pad=14)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.grid(axis="y", color="#e5e5e5", lw=1)
ax.set_axisbelow(True)
ax.tick_params(axis="y", labelsize=15)
fig.text(0.01, 0.01,
         "L4: each test chunk's own whole-run average. A40/L40: cluster monitor readings recorded in our log.\n"
         "The simulator runs on the CPU, so a faster GPU just waits more.",
         fontsize=13, color=GREY, ha="left", va="bottom", linespacing=1.3)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(OUT2, dpi=200, facecolor="white")
print("wrote", OUT2)
