"""Blind baselines for the 2026-10-19 feasibility tally (n = 4 partners).

Reads the four tally JSONs, ranks the partners L, S, G, R by each frozen-feature
predictor, and places each Spearman rho (against the paper's Table 10 order
L > S > G > R) on the distribution of rho over all 24 random orderings.
Descriptive only. Pure Python plus numpy. Run: python -I src/tally_baselines.py
"""
import json
from itertools import permutations
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "experiments/2026-10-19-feasibility-tally/results"
FILES = {"L": "siglip2", "R": "c-radiov4", "S": "sam3", "G": "vggt-omega"}
PAPER_ORDER = ["L", "S", "G", "R"]  # liu2026what Table 10, best to worst (D+B mean)
PAPER_DELTA = {"L": 1.83, "S": 0.49, "G": 0.40, "R": -5.41}  # Table 10 mean minus D alone (84.0)
# Overlap label, fixed before any D+R proxy result; rule and reasoning are in the memo.
LABEL = {"L": 1, "S": 0, "G": 0, "R": -1}  # +1 helpful, 0 neutral, -1 harmful


def avg_ranks(values):
    """Rank 1 = largest value; ties get the average rank."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = np.array(avg_ranks(a)), np.array(avg_ranks(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def load():
    out = {}
    for k, name in FILES.items():
        with open(RES / f"tally-libero_goal-dinov3-{name}.json") as f:
            out[k] = json.load(f)
    return out


def predictors(data, ft):
    """Return {predictor name: {partner: value}}, higher = predicted better partner."""
    p = {"solo R2(B)": {}, "gain(B|D)": {}, "gain(D|B)": {}, "CKA agentview (lower=better)": {},
         "CKA wrist (lower=better)": {}, "CKA mean of cams (lower=better)": {}}
    for k, d in data.items():
        n = FILES[k]
        p["solo R2(B)"][k] = d["probes"][f"{n}|{ft}"]["r2_mean"]
        p["gain(B|D)"][k] = d["pairs"][f"dinov3|{n}|{ft}"]["gain_B_given_A"]
        p["gain(D|B)"][k] = d["pairs"][f"{n}|dinov3|{ft}"]["gain_B_given_A"]
        a = d["cka"][f"dinov3|{n}|{ft}|agentview_rgb"]["debiased"]
        w = d["cka"][f"dinov3|{n}|{ft}|eye_in_hand_rgb"]["debiased"]
        p["CKA agentview (lower=better)"][k] = -a
        p["CKA wrist (lower=better)"][k] = -w
        p["CKA mean of cams (lower=better)"][k] = -(a + w) / 2
        # identity checks: gain(D|B) = R2(D+B) - R2(B); gain(B|D) = R2(D+B) - R2(D)
        # the C-RADIOv4 JSON names its concat probe "c-radiov4+dinov3"; the others "dinov3+X"
        key = f"dinov3+{n}|{ft}" if f"dinov3+{n}|{ft}" in d["probes"] else f"{n}+dinov3|{ft}"
        r_db = d["probes"][key]["r2_mean"]
        assert abs((r_db - p["solo R2(B)"][k]) - p["gain(D|B)"][k]) < 1e-9
        assert abs((r_db - d["probes"][f"dinov3|{ft}"]["r2_mean"]) - p["gain(B|D)"][k]) < 1e-9
    return p


def main():
    data = load()
    partners = PAPER_ORDER
    paper_vals = [PAPER_DELTA[b] for b in partners]
    paper_rank = avg_ranks(paper_vals)

    print("== (b) Random ranking, n = 4, against paper order L > S > G > R ==")
    dist = {}
    rows = []
    for perm in permutations(range(1, 5)):  # perm[i] = rank given to partners[i]
        rho = spearman([-r for r in perm], [-r for r in paper_rank])
        rows.append((perm, rho))
        dist[round(rho, 4)] = dist.get(round(rho, 4), 0) + 1
    print("rho    count/24")
    for r in sorted(dist, reverse=True):
        cum = sum(c for x, c in dist.items() if x >= r)
        print(f"{r:+.1f}  {dist[r]:2d}   cum P(rho >= this) = {cum}/24")
    rho_all = np.array([r for _, r in rows])
    print(f"mean rho = {rho_all.mean():+.3f}")
    print("\nAll 24 orderings (best to worst; rho vs paper):")
    for perm, rho in rows:
        order = [b for _, b in sorted(zip(perm, partners))]
        print(" > ".join(order), f"  rho={rho:+.1f}")
    r_last = sum(1 for perm, _ in rows if perm[partners.index("R")] == 4)
    print(f"\nOrderings with R last: {r_last}/24 = {r_last/24:.3f}  (arithmetic: 3!/4! = 6/24)")

    def place(rho):
        ge = int((rho_all >= rho - 1e-9).sum())
        le = int((rho_all <= rho + 1e-9).sum())
        return f"P(random rho >= obs) = {ge}/24, P(random rho <= obs) = {le}/24"

    for ft in ("global", "mean"):
        print(f"\n== (c) Predictors, feature type: {ft} ==")
        preds = predictors(data, ft)
        print("Raw values (CKA shown as debiased CKA):")
        for name, v in preds.items():
            sgn = -1 if "CKA" in name else 1
            print(f"  {name:34s}", "  ".join(f"{b}={sgn*v[b]:+.4f}" for b in partners))
        print("\nOrdering (best to worst) and Spearman rho vs paper order L > S > G > R:")
        for name, v in preds.items():
            vals = [v[b] for b in partners]
            ranks = avg_ranks(vals)
            order = " > ".join(b for _, b in sorted(zip(ranks, partners)))
            rho = spearman(vals, paper_vals)
            r_pos = ranks[partners.index("R")]
            print(f"  {name:34s} {order}   rho={rho:+.2f}   R rank={r_pos:g}/4   {place(rho)}")

    lv = [LABEL[b] for b in partners]
    rho = spearman(lv, paper_vals)
    print("\n== Overlap label (+1 helpful, 0 neutral, -1 harmful) ==")
    print("  labels:", "  ".join(f"{b}={LABEL[b]:+d}" for b in partners),
          f"  rho vs paper = {rho:+.2f} (label has ties; average ranks)   {place(rho)}")


main()
