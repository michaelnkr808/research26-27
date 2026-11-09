"""DINOv3 global features of the STORED frames, train and val demos of all ten libero_goal tasks, computed on the Mac (pf-val diagnosis, all-task part).

Same encoder code as the tally (extract_features.py dinov3 loader; rotate 180, bicubic to 224, CLS token). The split is made from demo names
with diagnostics.demo_split BEFORE any pixel array is read; test demos' images are never read. Writes one npz per task into --out:
keys agentview_rgb, eye_in_hand_rgb ([N,1024] float32), demo (int), j (int), split (0 train, 1 val). Order: all val first (pass 1), then train (pass 2).
Resumable: an existing file for a pass is skipped. For the stove task, an existing stored_all.npz from pf_val_diag_features.py can be converted with --from-stove.
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import diagnostics as D
import pf_val_diag_features as F

CAMS = F.CAMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--stove-npz", default="", help="reuse stored_all.npz of the stove task (same code, train+val demos)")
    ap.add_argument("--train-stride", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    enc, torch = F.make_encoder(args.device)
    files = sorted(Path(args.raw).glob("*.hdf5"))
    t0 = time.time()
    for part in ("val", "train"):
        for p in files:
            dst = out / f"{p.stem}.{part}.npz"
            if dst.exists():
                continue
            if args.stove_npz and p.stem.startswith("turn_on_the_stove"):
                z = np.load(args.stove_npz)
                sel = (z["demo"] >= 30) if part == "val" else (z["demo"] < 30)
                np.savez(dst, **{c: z[c][sel] for c in CAMS}, demo=z["demo"][sel], j=z["j"][sel])
                continue
            with h5py.File(p, "r") as f:
                names = sorted(f["data"].keys(), key=F.dnum)
                sp = D.demo_split(names)                                   # before any pixel read
                use = [d for d in names if sp[d] == part]
                assert all(sp[d] != "test" for d in use)
                feats = {c: [] for c in CAMS}
                dem, jj = [], []
                for d in use:
                    n = len(f["data"][d]["actions"])
                    stride = args.train_stride if part == "train" else 1
                    ix = np.arange(0, n, stride)
                    for c in CAMS:
                        fr = f["data"][d]["obs"][c][:][ix]
                        feats[c].append(F.encode(enc, torch, fr))
                    dem += [F.dnum(d)] * len(ix)
                    jj += ix.tolist()
            np.savez(dst, **{c: np.concatenate(feats[c]) for c in CAMS}, demo=np.array(dem), j=np.array(jj))
            print(f"{p.stem} {part} {len(dem)} frames, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
