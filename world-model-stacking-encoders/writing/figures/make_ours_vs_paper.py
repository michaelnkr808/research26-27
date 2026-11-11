# Slide figure: our proxy WAM vs the paper's WAM (liu2026what), facts only.
# Run with /Users/michaelr808/Desktop/research/research26-27/vis-aria-depth/.venv/bin/python
# Sources (checked 2026-11-11):
#   ours: src/proxy_wam.py defaults (--layers 8, --epochs 10, --batch-size 128), 31,647,943 params
#         (proxy PLAN line 287, training logs); success: proxy LOG 2026-11-09 (t) (8.3 / 7.2 LIBERO-Plus),
#         clean 8/40 and 7/40 counted by the planning lead 2026-11-11 from results/full-benchmark/rollouts/.
#   paper: arXiv 2610.13024v1 Sec 3.1 ("video expert initialized from Wan 2.2"), App B.2 ("each of the
#         30 action-expert blocks"), App C.1 Table 7 (LIBERO: 10 epochs, batch 128, 1,712 demos);
#         parameter count and Wan 2.2 variant not stated. Table 1/10 CSV: DINOv3 84.0 LIBERO-Plus.
#         Clean LIBERO ~97-99% across single-encoder models (paper note, App D Table 8).
#   Wan2.2-TI2V-5B config.json: num_layers 30, dim 3072 (footnote only; paper does not name the variant).
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "2026-11-11-ours-vs-paper.png")

OURS = "#d4572a"
PAPER = "#2b6cb0"
SAME = "#8a8a8a"
INK = "#1f1f1f"

rows = [
    # label, ours, paper, differs?
    ("Starting point", "Random weights\n(trained from scratch)", "Video part starts from Wan 2.2,\na large pretrained video model", True),
    ("Depth", "8 layers", "30 action-expert blocks;\nvideo part not stated*", True),
    ("Size", "About 32 million parameters", "Not stated in the paper*", True),
    ("Training", "10 epochs, batch 128", "10 epochs, batch 128", False),
    ("Training data", "Same 1,712 LIBERO demos", "Same 1,712 LIBERO demos", False),
]

fig = plt.figure(figsize=(13.33, 7.5), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 13.33)
ax.set_ylim(0, 7.5)
ax.axis("off")

ax.text(0.5, 7.0, "Our copy vs. the paper's model", fontsize=24, color=INK, weight="bold", va="center")

# column headers
x_lab, x_ours, x_paper = 0.5, 2.5, 5.45
col_w = 2.75
ax.text(x_ours + col_w / 2, 6.35, "Ours", fontsize=17, color=OURS, weight="bold", ha="center")
ax.text(x_paper + col_w / 2, 6.35, "Paper (Liu et al. 2026)", fontsize=17, color=PAPER, weight="bold", ha="center")

y = 5.75
h = 0.82
for label, o, p, differs in rows:
    ax.text(x_lab, y - h / 2 + 0.05, label, fontsize=13.5, color=INK, va="center", weight="bold")
    for x, txt, col in ((x_ours, o, OURS), (x_paper, p, PAPER)):
        c = col if differs else SAME
        box = FancyBboxPatch((x, y - h + 0.06), col_w, h - 0.12, boxstyle="round,pad=0.02,rounding_size=0.08",
                             fc=c, alpha=0.12 if differs else 0.08, ec=c, lw=1.4)
        ax.add_patch(box)
        ax.text(x + col_w / 2, y - h / 2 + 0.03, txt, fontsize=12, color=INK, ha="center", va="center")
    if not differs:
        ax.text(x_paper + col_w + 0.15, y - h / 2 + 0.03, "same", fontsize=12, color=SAME, va="center", style="italic")
    y -= h

# success bars on the right
bx0, bw = 10.0, 2.85
ax.text(bx0 + bw / 2, 6.35, "Tasks finished", fontsize=17, color=INK, weight="bold", ha="center")

def bars(ytop, title, ours_v, paper_v, ours_lab, paper_lab):
    ax.text(bx0, ytop, title, fontsize=12.5, color=INK, weight="bold", va="center")
    for i, (v, lab, col, name) in enumerate(((ours_v, ours_lab, OURS, "Ours"), (paper_v, paper_lab, PAPER, "Paper"))):
        yy = ytop - 0.5 - i * 0.55
        ax.add_patch(FancyBboxPatch((bx0, yy - 0.18), bw, 0.36, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc="#ececec", ec="none"))
        ax.add_patch(FancyBboxPatch((bx0, yy - 0.18), max(bw * v / 100, 0.06), 0.36,
                                    boxstyle="round,pad=0,rounding_size=0.05", fc=col, ec="none"))
        ax.text(bx0 - 0.1, yy, name, fontsize=11, color=col, ha="right", va="center", weight="bold")
        inside = v > 45
        ax.text(bx0 + bw * v / 100 + (-0.08 if inside else 0.08), yy, lab, fontsize=11.5,
                color="white" if inside else INK, ha="right" if inside else "left", va="center", weight="bold")

bars(5.6, "Disturbed scenes (LIBERO-Plus)", 7.7, 84.0, "about 8%", "84%")
bars(3.85, "Clean scenes (LIBERO)", 18.75, 98.0, "just under 1 in 5", "near perfect")

ax.text(bx0, 2.5, "Both: DINOv3 added. Ours: average of 2 runs,\nclean from only 40 tries each. Paper clean:\nabout 97-99% across its single-encoder models.",
        fontsize=9, color="#555555", va="top")

ax.text(0.5, 1.35,
        "Same training length and data. The big difference is the starting point:\n"
        "the paper's model has already learned from huge amounts of video\nbefore it sees a robot. Ours starts from nothing.",
        fontsize=13, color=INK, va="center")
ax.text(0.5, 0.45,
        "* The paper does not say which Wan 2.2 model it used or how many parameters its model has. Wan 2.2's 5B model has 30 layers.\n"
        "Sources: Liu et al. 2026 (arXiv 2610.13024) Sec. 3.1, App. B.2, App. C.1 Table 7, Tables 1 and 10; our training code and evaluation logs.",
        fontsize=9.5, color="#666666", va="center")

fig.savefig(OUT, dpi=150, facecolor="white")
print(OUT)
