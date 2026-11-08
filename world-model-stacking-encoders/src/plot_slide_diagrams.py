# Versions used: Python 3.11.14, matplotlib 3.11.2 (also printed at run time).
# Run with /Users/michaelr808/Desktop/research/research26-27/vis-aria-depth/.venv/bin/python
#
# Three slide diagrams (no data plotted; schematic only). Same look as plot_training_curve.py.
#   1. noise-check ruler   2. combining-step grids   3. encoder layers
# Every number on the figures comes from facts checked by the lead against repo files.
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "writing", "figures")

BLUE, BLUE2, ORANGE, INK, GREY = "#1f5fa8", "#6fa3d8", "#d9480f", "#1a1a1a", "#555555"
BLUE_BG, ORANGE_BG, LIGHT = "#dbe8f6", "#fbe3d6", "#e5e5e5"
plt.rcParams.update({"font.size": 17, "font.family": "sans-serif"})
print("python", sys.version.split()[0], "matplotlib", matplotlib.__version__)


def new(w, h, title, sub=None):
    fig = plt.figure(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes((0.02, 0.14, 0.96, 0.70))
    ax.axis("off")
    fig.text(0.02, 0.95, title, fontsize=24, fontweight="bold", color=INK, ha="left", va="top")
    if sub:
        fig.text(0.02, 0.885, sub, fontsize=16, color=GREY, ha="left", va="top")
    return fig, ax


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", os.path.abspath(path), "size (w x h px):", Image.open(path).size)


def arrow(ax, p, q, color=GREY, lw=2.5, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=22, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------- 1. noise-check ruler
fig, ax = new(11, 6.9, "What the noise check compares")
ax.set_position((0.02, 0.17, 0.96, 0.67))
ax.set_xlim(-0.3, 7.3)
ax.set_ylim(-3.2, 7.4)
Y0, H = 0.0, 0.5
ax.add_patch(Rectangle((0, Y0 - H), 2.7, 2 * H, color=BLUE_BG, lw=0))
ax.add_patch(Rectangle((2.7, Y0 - H), 3.5, 2 * H, color=ORANGE_BG, lw=0))
ax.plot([0, 6.2], [Y0, Y0], color=INK, lw=3, solid_capstyle="butt")
for t in range(0, 7):
    ax.plot([t, t], [Y0 - 0.18, Y0 + 0.18], color=INK, lw=2)
    ax.text(t, Y0 - 0.65, str(t), ha="center", va="top", fontsize=16, color=INK)
ax.text(3.1, Y0 - 1.25, "Gap in success rate (percentage points)", ha="center", va="top", fontsize=16, color=GREY)
# zone labels
ax.text(1.35, -2.0, "Luck is small:\ngo ahead and train\nthe other 13 models", ha="center", va="top",
        fontsize=17, fontweight="bold", color=BLUE, linespacing=1.25)
ax.text(4.6, -2.0, "Luck is as big as the harm:\nrethink the plan", ha="center", va="top",
        fontsize=17, fontweight="bold", color=ORANGE, linespacing=1.25)
# marks
for x, col, txt in ((2.7, BLUE, "Our “really harmful” bar\n(half the paper’s drop)"),
                    (5.4, ORANGE, "Paper’s drop: DINOv3 + C-RADIOv4\nvs DINOv3 alone (78.6 vs 84.0)")):
    ax.plot([x, x], [Y0 + H, 2.7], color=col, lw=3)
    ax.scatter([x], [Y0], s=170, color=col, zorder=5, edgecolor="white", linewidth=2)
    ax.text(x + (0.3 if x < 3 else 0), 2.9, txt, ha="right" if x < 3 else "center", va="bottom", fontsize=16, fontweight="bold", color=col, linespacing=1.25)
ax.text(2.6, 2.55, "2.7", ha="right", va="top", fontsize=16, color=BLUE, fontweight="bold")
ax.text(5.5, 2.55, "5.4", ha="left", va="top", fontsize=16, color=ORANGE, fontweight="bold")
# unknown marker: dashed box, deliberately not tied to any position on the ruler
ax.add_patch(FancyBboxPatch((0.5, 5.2), 5.3, 1.7, boxstyle="round,pad=0.05,rounding_size=0.2", fill=False,
                            ec=GREY, lw=2.5, ls=(0, (5, 4))))
ax.text(1.0, 6.05, "?", ha="center", va="center", fontsize=40, fontweight="bold", color=GREY)
ax.text(1.55, 6.05, "Gap between our two identical copies:\nbeing measured now", ha="left", va="center",
        fontsize=17, color=INK, linespacing=1.3)
fig.text(0.02, 0.02, "Also: if either copy almost never succeeds, the comparison means\nnothing and we decide what to do.",
         fontsize=16, color=GREY, ha="left", va="bottom", linespacing=1.3)
save(fig, "2026-11-07-noise-check-ruler.png")

# ---------------------------------------------------------------- 2. combining grids
# Axes units are inches (1 unit = 1 in), so text sizes are easy to reason about.
# Figure is 9.5 in wide: scaled to ~5 in on a slide, 18 pt text stays at about 9.5 pt.
fig = plt.figure(figsize=(9.5, 10.2), dpi=200)
fig.patch.set_facecolor("white")
ax = fig.add_axes((0.02, 1.35 / 10.2, 0.96, 7.9 / 10.2))
ax.axis("off")
fig.text(0.02, 0.975, "How our copy stacks the encoders", fontsize=26, fontweight="bold", color=INK, ha="left", va="top")
ax.set_xlim(0, 9.12)
ax.set_ylim(0, 7.9)
ax.set_aspect("equal")
C_VAE, C_DINO, C_RAD = "#8a8a8a", BLUE, ORANGE


def lattice(x, y, n, col, size=1.3):
    ax.add_patch(Rectangle((x, y), size, size, fill=False, ec=col, lw=2.2))
    step = size / n
    for i in range(1, n):
        ax.plot([x + i * step] * 2, [y, y + size], color=col, lw=0.6 if n > 14 else 1.0)
        ax.plot([x, x + size], [y + i * step] * 2, color=col, lw=0.6 if n > 14 else 1.0)


def label(y_top, name, g, nums, col):
    ax.text(1.5, y_top, name, fontsize=19, fontweight="bold", color=col, va="top")
    ax.text(1.5, y_top - 0.46, g, fontsize=18, color=INK, va="top")
    ax.text(1.5, y_top - 0.88, nums, fontsize=18, color=INK, va="top")


lattice(0, 5.9, 14, C_VAE)
label(7.2, "VAE (reference grid)", "14 x 14 spots", "48 numbers per spot", C_VAE)
lattice(0, 3.7, 14, C_DINO)
label(5.0, "DINOv3", "14 x 14 spots", "1,024 numbers per spot", C_DINO)
lattice(0, 2.0, 28, C_RAD)
label(3.3, "C-RADIOv4", "28 x 28 spots (finer)", "1,152 numbers per spot", C_RAD)
# shrink step for C-RADIOv4
arrow(ax, (0.65, 1.92), (0.65, 1.5), C_RAD)
ax.text(1.5, 1.72, "shrink to 14 x 14", fontsize=19, fontweight="bold", color=C_RAD, va="center")
lattice(0, 0.1, 14, C_RAD)
# arrows into the stack
arrow(ax, (4.8, 6.7), (5.9, 5.45), C_VAE)
arrow(ax, (4.8, 4.4), (5.9, 4.5), C_DINO)
ax.plot([1.5, 5.5], [0.75, 0.75], color=C_RAD, lw=2.5, solid_capstyle="butt")
ax.plot([5.5, 5.5], [0.75, 3.1], color=C_RAD, lw=2.5, solid_capstyle="butt")
arrow(ax, (5.5, 3.1), (5.9, 3.1), C_RAD)
# stack block (segment heights only roughly suggest size, not to scale)
for (y0, h, col, t) in ((5.2, 0.4, C_VAE, "VAE  48"), (3.9, 1.25, C_DINO, "DINOv3  1,024"), (2.4, 1.45, C_RAD, "C-RADIOv4  1,152")):
    ax.add_patch(Rectangle((6.0, y0), 3.1, h, fc=col, ec="white", lw=2.5))
    ax.text(7.55, y0 + h / 2, t, ha="center", va="center", fontsize=16, fontweight="bold", color="white")
ax.text(6.0, 2.15, "At each of the\n14 x 14 spots:\n48 + 1,024 + 1,152\n= 2,224 numbers,\ninto the robot model", ha="left", va="top",
        fontsize=16, fontweight="bold", color=INK, linespacing=1.3)
ax.text(6.0, 5.85, "Before stacking, DINOv3's\nand C-RADIOv4's numbers\nare rescaled to a similar\nsize (the VAE's are not)",
        ha="left", va="bottom", fontsize=15, color=GREY, linespacing=1.25)
fig.text(0.02, 0.02, "Our copy. The paper doesn't say how it\nlines up or rescales the grids.\nBlock sizes are not to scale. Numbers are per camera image.",
         fontsize=17, color=GREY, ha="left", va="bottom", linespacing=1.3)
save(fig, "2026-11-07-combining-grids.png")

# ---------------------------------------------------------------- 3. encoder layers
fig, ax = new(8, 7.2, "We only use each encoder's final layer")
ax.set_xlim(0, 100)
ax.set_ylim(0, 48)
bh, gap, y = 5.4, 1.6, 9
ax.add_patch(Rectangle((4, 0.5), 38, 5.2, fc=LIGHT, ec=GREY, lw=2))
ax.text(23, 3.1, "Camera image", ha="center", va="center", fontsize=18, color=INK)
arrow(ax, (23, 5.9), (23, y - 0.1), GREY)
slots = ["earlier layer", "dots", "earlier layer", "final layer"]
ys = []
for i, t in enumerate(slots):
    yy = y + 1.2 + i * (bh + gap)
    ys.append(yy)
    if t == "dots":
        ax.text(23, yy + bh / 2, "⋮", ha="center", va="center", fontsize=34, color=GREY)
        ax.text(27, yy + bh / 2, "many more", ha="left", va="center", fontsize=16, color=GREY)
        continue
    last = t == "final layer"
    ax.add_patch(FancyBboxPatch((4, yy), 38, bh, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=ORANGE if last else "#f2f2f2", ec=ORANGE if last else GREY, lw=3 if last else 2,
                                ls="-" if last else (0, (4, 3))))
    ax.text(23, yy + bh / 2, t, ha="center", va="center", fontsize=18, fontweight="bold" if last else "normal",
            color="white" if last else GREY)
top = ys[-1] + bh
arrow(ax, (23, top + 0.2), (23, top + 3.6), ORANGE)
ax.text(23, top + 4.2, "Features out", ha="center", va="bottom", fontsize=18, color=INK, fontweight="bold")
ax.plot([45, 45], [ys[0], ys[-2] + bh], color=GREY, lw=3)
ax.text(48, (ys[0] + ys[-2] + bh) / 2, "Not explored yet:\nwhat do they add?", ha="left", va="center",
        fontsize=20, fontweight="bold", color=GREY, linespacing=1.3)
ax.plot([45, 45], [ys[-1], top], color=ORANGE, lw=3)
ax.text(48, ys[-1] + bh / 2, "Used by the paper\nand by us", ha="left", va="center",
        fontsize=20, fontweight="bold", color=ORANGE, linespacing=1.3)
save(fig, "2026-11-07-encoder-layers.png")
