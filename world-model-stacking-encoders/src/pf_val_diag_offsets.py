"""Frame-offset check, stove task, val demos (pf-val diagnosis): mean absolute pixel difference between stored image j+off and re-render j,
for off in -2..2, per camera, averaged over all val demos. Offset 0 is the pairing used everywhere else (frame j = render of states[j+1]).
Writes offsets-stove.json."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pf_val_diag_pixels import CAMS, load_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stored", required=True)
    ap.add_argument("--rerender", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    P = load_pairs(args.stored, args.rerender)
    import h5py
    rep = {}
    with h5py.File(args.stored, "r") as fs, h5py.File(args.rerender, "r") as fr:
        for c in CAMS:
            tot = {o: [] for o in range(-2, 3)}
            for d in sorted(fr["data"].keys(), key=lambda k: int(k.split("_")[-1])):
                r = fr["data"][d]["obs"][c][:].astype(np.float32)
                s = fs["data"][d]["obs"][c][:].astype(np.float32)          # n frames; re-render has n-1
                n = len(r)
                for o in range(-2, 3):
                    lo, hi = max(0, -o), min(n, n - o)                     # re-render index j in [lo, hi) pairs with stored j+o
                    tot[o].append(np.abs(s[lo + o:hi + o] - r[lo:hi]).mean())
            rep[c] = {str(o): float(np.mean(v)) for o, v in tot.items()}
    (Path(args.out) / "offsets-stove.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
