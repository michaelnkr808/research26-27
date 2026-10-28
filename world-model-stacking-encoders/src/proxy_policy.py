"""Proxy policy on cached frozen-encoder features (decision 003; experiments/2026-10-27-proxy-calibration/PLAN.md).

A small transformer that maps the current frame to a 32-step action chunk and also predicts every source's future
features, so each encoder is both an input and a prediction target, as in liu2026what (App. E.1, Eq. 9).

Streams (one per source, all on the same 2 cams x g x g token grid):
  - pixels: raw frames, rotated 180 deg, block-averaged to (g * PATCH)^2 and cut into g x g patches of PATCH^2 * 3
    values. Present in every config, as the paper's VAE latents are (`--no-base-pixels` drops it).
  - one stream per encoder in the config: the cached `grid` features from extract_features.py.
Fusion is channel-wise concatenation (the paper's default): per position, each stream is projected to width W, the
projections are concatenated along channels and mapped to the model width H.

Per sample (frame t of an episode):
  inputs   every stream at t (z-scored per channel on train), proprio (ee pose 6 + gripper 2, z-scored), and a frozen
           text embedding of the instruction (same text encoder for every config; LIBERO-Plus rewrites instructions,
           so a task id would not do).
  targets  actions t .. t + 31 (min/max-normalized to [-1, 1]); every stream at t + 4, 8, ..., 32 (z-scored).
           Steps past the episode end are masked out of the loss (Fast-WAM pads them; we mask).
  loss     action MSE + sum over streams of that stream's feature MSE, each a per-source mean, weight 1.0 each.
Deterministic regression stands in for the paper's flow matching (proxy simplification, logged in PLAN).

Optimizer as paper Table 6: AdamW, lr 1e-4 -> 1e-6 cosine, 5% linear warmup, weight decay 0.01, betas (0.9, 0.95),
grad clip 1.0. The seed sets the model init and the data order; the data order doesn't depend on the config.

Example:
  python proxy_policy.py --feats /work/feats --suite libero_goal --raw /work/data/libero_goal \
      --config dinov3+c-radiov4 --seed 0 --epochs 10 --out /work/runs/d+r-s0
"""

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

import diagnostics as D

HORIZON = 32                      # action chunk (paper Table 6)
OFFSETS = list(range(4, 33, 4))   # future feature frames t+4 .. t+32 (paper Table 6: offsets 0, 4, ..., 32)
PATCH = 8                         # pixel stream: PATCH x PATCH pixels per token
PIXELS = "pixels"


def git_commit():
    return D.git_commit()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def pixel_tokens(frames, g):
    """(T, H, W, 3) uint8 frames, already upright -> (T, g * g, PATCH^2 * 3) float32 in [0, 1], row-major positions."""
    t, h, w, _ = frames.shape
    s = g * PATCH
    assert h % s == 0 and w % s == 0, f"frame {h}x{w} is not a multiple of {s}"
    x = frames.reshape(t, s, h // s, s, w // s, 3).mean((2, 4), dtype=np.float32) / 255.0  # block average to s x s
    x = x.reshape(t, g, PATCH, g, PATCH, 3).transpose(0, 1, 3, 2, 4, 5)
    return np.ascontiguousarray(x.reshape(t, g * g, PATCH * PATCH * 3))


class Episodes:
    """All frames of all episodes, concatenated. streams[name]: (N, P, d) float16, P = cams x g x g positions."""

    def __init__(self, streams, actions, proprio, task, ep_index, split, text):
        self.streams, self.actions, self.proprio, self.task = streams, actions, proprio, task
        self.text = text                                                  # (n_tasks, d_text) frozen instruction embeddings
        self.ep_index, self.split = ep_index, split                       # per frame: episode id, "train"/"val"
        n = len(actions)
        starts = np.r_[0, np.flatnonzero(np.diff(ep_index)) + 1]
        ends = np.r_[starts[1:], n]
        self.ep_end = np.repeat(ends, ends - starts)                      # per frame: index one past its episode
        for k, v in streams.items():
            assert len(v) == n, (k, len(v), n)

    def __len__(self):
        return len(self.actions)

    def windows(self, t):
        """Action indices (B, HORIZON) and future indices (B, len(OFFSETS)), clipped to the episode, with masks."""
        end = self.ep_end[t][:, None]
        a = t[:, None] + np.arange(HORIZON)[None]
        f = t[:, None] + np.array(OFFSETS)[None]
        return np.minimum(a, end - 1), a < end, np.minimum(f, end - 1), f < end


def hash_text_embedding(texts, dim=64):
    """SMOKE TESTS ONLY: a fixed random unit vector per instruction (seeded by its bytes). Carries no meaning, so
    reworded LIBERO-Plus instructions get unrelated vectors. Real runs pass --text-emb from a frozen text encoder."""
    out = []
    for t in texts:
        v = np.random.default_rng(list(t.encode())).standard_normal(dim)
        out.append(v / np.linalg.norm(v))
    return np.array(out, np.float32)


def load_text_embedding(path, texts):
    """(n, d) rows for texts from an .npz holding `texts` (str array) and `emb` (n, d)."""
    z = np.load(path)
    table = dict(zip(z["texts"].tolist(), z["emb"].astype(np.float32)))
    missing = [t for t in texts if t not in table]
    assert not missing, f"no text embedding for {missing}"
    return np.stack([table[t] for t in texts])


def load_libero_hdf5(feats_root, suite, raw_dir, encoders, val_frac=0.0, max_demos=None, g=4, rotate180=True,
                     text_emb=None):
    """Episodes from yifengzhu-hf LIBERO HDF5 files plus extract_features.py grid caches.

    The val split holds out the last val_frac of each file's demos (sorted by index), as diagnostics.demo_split does.
    rotate180 applies to the raw frames of the pixel stream; it must match the cache's own `rotate180` (checked).
    """
    raw_files = sorted(Path(raw_dir).glob("*.hdf5"))
    assert raw_files, f"no LIBERO files in {raw_dir}"
    cols = {k: [] for k in [PIXELS] + list(encoders)}
    actions, proprio, task, ep_index, split, tasks = [], [], [], [], [], []
    meta = {}
    for ti, raw in enumerate(raw_files):
        fe = {e: h5py.File(Path(feats_root) / e / suite / (raw.stem + ".h5"), "r") for e in encoders}
        for e, f in fe.items():
            m = json.loads(f.attrs["meta"])
            assert m["rotate180"] == rotate180 and m["grid"] == g, f"{e} cache: rotate180/grid differ from this run"
            meta.setdefault(e, m)
        with h5py.File(raw, "r") as fr:
            tasks.append(json.loads(fr["data"].attrs["problem_info"])["language_instruction"])
            demos = sorted(fr["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            demos = [d for d in demos if all(d in f for f in fe.values())][:max_demos]
            n_val = round(len(demos) * val_frac)
            for di, d in enumerate(demos):
                obs = fr["data"][d]["obs"]
                n = len(fr["data"][d]["actions"])
                px = []
                for c in D.CAMS:
                    fr_c = obs[c][:]
                    if rotate180:
                        fr_c = fr_c[:, ::-1, ::-1]
                    px.append(pixel_tokens(np.ascontiguousarray(fr_c), g))
                cols[PIXELS].append(np.concatenate(px, 1).astype(np.float16))
                for e, f in fe.items():
                    gr = [f[d][c]["grid"][:] for c in D.CAMS]
                    assert all(len(x) == n for x in gr), f"{raw.name} {d} {e}: feature length != {n} actions"
                    cols[e].append(np.concatenate([x.reshape(n, g * g, -1) for x in gr], 1).astype(np.float16))
                actions.append(fr["data"][d]["actions"][:].astype(np.float32))
                proprio.append(np.concatenate([obs["ee_states"][:], obs["gripper_states"][:]], 1).astype(np.float32))
                task += [ti] * n
                ep_index += [len(actions) - 1] * n
                split += ["val" if di >= len(demos) - n_val else "train"] * n
        for f in fe.values():
            f.close()
    text = load_text_embedding(text_emb, tasks) if text_emb else hash_text_embedding(tasks)
    ep = Episodes({k: np.concatenate(v) for k, v in cols.items()}, np.concatenate(actions), np.concatenate(proprio),
                  np.array(task), np.array(ep_index), np.array(split), text)
    return ep, tasks, meta


# ---------------------------------------------------------------------------
# Normalization (statistics from train frames only)
# ---------------------------------------------------------------------------

class Norm:
    def __init__(self, ep):
        tr = ep.split == "train"
        self.stream = {}
        for k, v in ep.streams.items():
            x = v[tr].astype(np.float64).reshape(-1, v.shape[-1])                # per channel, pooled over positions
            self.stream[k] = (x.mean(0).astype(np.float32), (x.std(0) + 1e-6).astype(np.float32))
        a = ep.actions[tr]
        self.a_lo, self.a_hi = a.min(0), a.max(0)
        p = ep.proprio[tr]
        self.p_mu, self.p_sd = p.mean(0), p.std(0) + 1e-6

    def feats(self, k, x):
        mu, sd = self.stream[k]
        return (x.astype(np.float32) - mu) / sd

    def actions(self, a):
        return 2 * (a - self.a_lo) / np.maximum(self.a_hi - self.a_lo, 1e-6) - 1

    def actions_inv(self, a):
        return (a + 1) / 2 * np.maximum(self.a_hi - self.a_lo, 1e-6) + self.a_lo

    def proprio(self, p):
        return (p - self.p_mu) / self.p_sd

    def state(self):
        return {"stream": {k: [m.tolist(), s.tolist()] for k, (m, s) in self.stream.items()},
                "a_lo": self.a_lo.tolist(), "a_hi": self.a_hi.tolist(),
                "p_mu": self.p_mu.tolist(), "p_sd": self.p_sd.tolist()}


def batch(ep, norm, t, device):
    """Tensors for the frames t: inputs, targets and masks."""
    ai, am, fi, fm = ep.windows(t)
    x = {k: torch.from_numpy(norm.feats(k, v[t])).to(device) for k, v in ep.streams.items()}
    y = {k: torch.from_numpy(norm.feats(k, v[fi])).to(device) for k, v in ep.streams.items()}  # (B, K, P, d)
    return {"x": x, "y": y, "fmask": torch.from_numpy(fm).to(device),
            "proprio": torch.from_numpy(norm.proprio(ep.proprio[t])).to(device),
            "text": torch.from_numpy(ep.text[ep.task[t]]).to(device),
            "act": torch.from_numpy(norm.actions(ep.actions[ai]).astype(np.float32)).to(device),
            "amask": torch.from_numpy(am).to(device)}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ProxyPolicy(nn.Module):
    def __init__(self, dims, n_pos, d_text, d_proprio=8, width=64, hidden=256, layers=4, heads=8, act_dim=7):
        super().__init__()
        self.names = list(dims)                                            # fixed stream order = concat order
        self.proj = nn.ModuleDict({k: nn.Linear(d, width) for k, d in dims.items()})
        self.fuse = nn.Linear(width * len(dims), hidden)                   # channel-wise concatenation
        self.pos = nn.Parameter(torch.randn(n_pos, hidden) * 0.02)
        self.proprio = nn.Linear(d_proprio, hidden)
        self.text = nn.Linear(d_text, hidden)
        self.queries = nn.Parameter(torch.randn(HORIZON, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, 4 * hidden, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(hidden)
        self.act_head = nn.Linear(hidden, act_dim)
        self.feat_head = nn.ModuleDict({k: nn.Linear(hidden, len(OFFSETS) * d) for k, d in dims.items()})
        self.dims, self.n_pos = dict(dims), n_pos

    def forward(self, x, proprio, text):
        b = proprio.shape[0]
        h = self.fuse(torch.cat([self.proj[k](x[k]) for k in self.names], -1)) + self.pos  # (B, P, H)
        tok = torch.cat([h, self.proprio(proprio)[:, None], self.text(text)[:, None],
                         self.queries.expand(b, -1, -1)], 1)
        z = self.norm(self.body(tok))
        zs, za = z[:, :self.n_pos], z[:, -HORIZON:]
        feats = {k: self.feat_head[k](zs).view(b, self.n_pos, len(OFFSETS), self.dims[k]).transpose(1, 2)
                 for k in self.names}                                       # (B, K, P, d)
        return self.act_head(za), feats


def losses(out, bt):
    """Action loss and per-stream feature losses (each a masked mean over its own elements), weight 1.0 each."""
    act, feats = out
    am = bt["amask"][..., None].float()
    la = (((act - bt["act"]) ** 2) * am).sum() / (am.sum() * act.shape[-1])
    fm = bt["fmask"][:, :, None, None].float()
    lf = {}
    for k, p in feats.items():
        y = bt["y"][k]
        lf[k] = (((p - y) ** 2) * fm).sum() / (fm.sum() * y.shape[2] * y.shape[3])
    return la, lf, la + sum(lf.values())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def lr_at(step, total, base=1e-4, floor=1e-6, warmup=0.05):
    w = max(1, int(total * warmup))
    if step < w:
        return base * (step + 1) / w
    q = (step - w) / max(1, total - w)
    return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * q))


def data_order(n_train, epochs, seed):
    """Index order into the train frames: one permutation per epoch, from the seed only (not the config)."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(n_train) for _ in range(epochs)])


def evaluate(model, ep, norm, rows, bs, device):
    model.eval()
    tot = {"action": 0.0}
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            bt = batch(ep, norm, rows[i:i + bs], device)
            la, lf, _ = losses(model(bt["x"], bt["proprio"], bt["text"]), bt)
            w = len(rows[i:i + bs]) / len(rows)
            tot["action"] += float(la) * w
            for k, v in lf.items():
                tot[k] = tot.get(k, 0.0) + float(v) * w
    model.train()
    return tot


def train(ep, seed, epochs, bs, device, log, model_kw=None, eval_every=0, max_steps=None, lr=1e-4):
    """lr decays to lr / 100 (paper: 1e-4 -> 1e-6)."""
    torch.manual_seed(seed)
    norm = Norm(ep)
    dims = {k: v.shape[-1] for k, v in ep.streams.items()}
    model = ProxyPolicy(dims, next(iter(ep.streams.values())).shape[1], ep.text.shape[1], **(model_kw or {})).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    tr = np.flatnonzero(ep.split == "train")
    va = np.flatnonzero(ep.split == "val")
    order = tr[data_order(len(tr), epochs, seed)]
    steps = len(order) // bs if max_steps is None else min(max_steps, len(order) // bs)
    sched = lambda s: lr_at(s, steps, base=lr, floor=lr / 100)
    hist = []
    t0 = time.time()
    for s in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = sched(s)
        bt = batch(ep, norm, order[s * bs:(s + 1) * bs], device)
        la, lf, loss = losses(model(bt["x"], bt["proprio"], bt["text"]), bt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % max(1, steps // 20) == 0 or s == steps - 1:
            rec = {"step": s, "lr": sched(s), "loss": float(loss), "action": float(la),
                   **{f"feat/{k}": float(v) for k, v in lf.items()}, "sec": time.time() - t0}
            if len(va) and (eval_every and (s % eval_every == 0 or s == steps - 1)):
                rec["val"] = evaluate(model, ep, norm, va, bs, device)
            hist.append(rec)
            log(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}))
    return model, norm, hist


def predict_actions(model, norm, x_raw, proprio_raw, text, device):
    """Action chunk (B, HORIZON, 7) in env units, from raw (un-normalized) current-frame streams. For rollouts."""
    model.eval()
    with torch.no_grad():
        x = {k: torch.from_numpy(norm.feats(k, v)).to(device) for k, v in x_raw.items()}
        p = torch.from_numpy(norm.proprio(proprio_raw).astype(np.float32)).to(device)
        act, _ = model(x, p, torch.as_tensor(text, dtype=torch.float32, device=device))
    return norm.actions_inv(act.cpu().numpy())


def config_encoders(config):
    """'none' -> []; 'dinov3+c-radiov4' -> ['dinov3', 'c-radiov4']."""
    return [] if config == "none" else config.split("+")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feats", required=True, help="root written by extract_features.py --out")
    p.add_argument("--suite", required=True)
    p.add_argument("--raw", required=True, help="dir with the suite's LIBERO HDF5 files")
    p.add_argument("--config", required=True, help="none, or encoders joined by +, e.g. dinov3+c-radiov4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None, help="cap on steps (smoke tests)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--val-frac", type=float, default=0.0, help="Fast-WAM's LIBERO config uses 0.0")
    p.add_argument("--max-demos", type=int, default=None, help="demos per file (smoke tests)")
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--no-base-pixels", action="store_true")
    p.add_argument("--text-emb", default=None, help=".npz with texts, emb (frozen text encoder); default: hash (smoke only)")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    t0 = time.time()
    log = lambda m: print(f"[proxy {time.time() - t0:6.0f}s] {m}", flush=True)
    encs = config_encoders(args.config)
    ep, tasks, meta = load_libero_hdf5(args.feats, args.suite, args.raw, encs, args.val_frac, args.max_demos,
                                       text_emb=args.text_emb)
    if not args.text_emb:
        log("WARNING: hash text embeddings (smoke only; carry no meaning)")
    if args.no_base_pixels:
        assert encs, "config none needs the pixel stream"
        del ep.streams[PIXELS]
    log(f"loaded {len(ep)} frames, {ep.ep_index.max() + 1} episodes, {len(tasks)} tasks, streams "
        + ", ".join(f"{k} {v.shape[1:]}" for k, v in ep.streams.items()) + f" in {time.time() - t0:.0f}s")
    model, norm, hist = train(ep, args.seed, args.epochs, args.batch_size, args.device, log,
                              {"hidden": args.hidden, "layers": args.layers}, args.eval_every, args.max_steps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"config": args.config, "streams": list(ep.streams), "encoders": {e: meta[e] for e in encs},
           "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "max_steps": args.max_steps,
           "val_frac": args.val_frac, "max_demos": args.max_demos, "hidden": args.hidden, "layers": args.layers,
           "text_emb": args.text_emb, "horizon": HORIZON, "offsets": OFFSETS, "patch": PATCH, "suite": args.suite,
           "tasks": tasks,
           "n_frames": len(ep), "git_commit": git_commit(), "device": args.device,
           "n_params": sum(q.numel() for q in model.parameters())}
    torch.save({"model": model.state_dict(), "norm": norm.state(), "config": cfg}, out / "model.pt")
    (out / "train.json").write_text(json.dumps({"config": cfg, "history": hist, "seconds": time.time() - t0}, indent=1))
    log(f"wrote {out}, {cfg['n_params']} params")


if __name__ == "__main__":
    main()
