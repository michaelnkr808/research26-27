"""Pure-numpy helpers for Step 2 (make the sound and check it). No RLR library, no simulator, no scene files.

Everything that decides a number in the direction check lives here, so it is unit-tested on the Mac
(test_step2lib.py). The scripts that need the rented machine (export_poses.py, render_audio.py) call these.

Frames and signs (PLAN.md, "Frames"):
- All audio geometry is in the InteriorGS ("GS") frame: right-handed, z up, metres. Literally: a viewer looking
  along +y has +x on the right; looking along -y, +x is on the left. (The dataset card's names "x right, y back"
  are not relied on; the side rule is fixed by GN-Bench's look-at rule, image right = forward x up, and
  test_step2lib.CameraAndHeading checks it through c3lib.look_at_view.) The scene mesh, the source (target box centre) and the listener are all in it.
  The listener pose of each step is the video camera's: eye = GN-Bench transform_from_world_to_GS(position),
  forward = transform_from_world_to_GS(position + world forward) - eye (exactly what GN-Bench's
  _render_observations does for the camera), horizontal part, normalised.
- left = up x forward, right = forward x up (poses.axes_from_forward). A source is on the listener's left when
  (source - eye) . left > 0. Azimuth is atan2(v . left, v . forward) in degrees: 0 ahead, +90 left, -90 right,
  +-180 behind.
- Binaural channel 0 is the left ear, channel 1 the right ear (route A check 1 follow-up: with the MIT KEMAR file
  a source at +90 degrees gave the left ear first in channel 0, LOG 2026-11-20).
- Interaural time difference (ITD) is reported as right-ear delay minus left-ear delay: positive means the left
  ear hears it first, which means the cue says "left".
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

LEFT, RIGHT, NONE = 1, -1, 0
UP = np.array([0.0, 0.0, 1.0])


# ----------------------------------------------------------------------------- geometry and sides
def listener_axes(forward) -> tuple:
    """(forward, left, right, up) unit vectors for a horizontal forward vector (z up, right-handed)."""
    f = np.asarray(forward, dtype=np.float64).reshape(3).copy()
    f[2] = 0.0
    n = float(np.linalg.norm(f))
    if n < 1e-9:
        raise ValueError("forward vector has no horizontal part")
    f /= n
    right = np.cross(f, UP)
    return f, -right, right, UP.copy()


def source_direction(eye, forward, source) -> dict:
    """Where the source is relative to the listener. lateral_deg is the angle between the direction to the source
    and the median plane (positive left): asin(v . left / |v|), with v the full 3D vector. In the horizontal plane
    |lateral| = 90 - | |azimuth| - 90 |."""
    eye = np.asarray(eye, dtype=np.float64)
    v = np.asarray(source, dtype=np.float64) - eye
    fwd, left, _right, up = listener_axes(forward)
    d = float(np.linalg.norm(v))
    x_f, x_l, x_u = float(v @ fwd), float(v @ left), float(v @ up)
    lat = math.degrees(math.asin(max(-1.0, min(1.0, x_l / d)))) if d > 0 else 0.0
    side = LEFT if x_l > 0 else (RIGHT if x_l < 0 else NONE)
    return {
        "distance_m": d,
        "horizontal_distance_m": float(math.hypot(x_f, x_l)),
        "azimuth_deg": float(math.degrees(math.atan2(x_l, x_f))),
        "lateral_deg": lat,
        "elevation_deg": float(math.degrees(math.atan2(x_u, math.hypot(x_f, x_l)))) if d > 0 else 0.0,
        "true_side": side,
    }


def is_counted(los: bool, lateral_deg: float, min_lateral_deg: float) -> bool:
    """PLAN counting rule: line of sight from listener to source, and the source clearly to one side: at least
    min_lateral_deg from the median plane. In the horizontal plane that drops the front zone |azimuth| <
    min_lateral_deg and the rear zone |azimuth| > 180 - min_lateral_deg."""
    return bool(los) and abs(float(lateral_deg)) >= float(min_lateral_deg)


# ----------------------------------------------------------------------------- interaural time difference
def common_onset_index(ir: np.ndarray, rel: float) -> Optional[int]:
    """First sample where either ear reaches rel x the larger of the two ears' peak magnitudes."""
    a = np.abs(np.asarray(ir[:2], dtype=np.float64))
    peak = float(a.max()) if a.size else 0.0
    if not np.isfinite(peak) or peak <= 0:
        return None
    hit = (a >= rel * peak).any(axis=0)
    return int(np.argmax(hit))


def estimate_itd(ir: np.ndarray, sr: float, onset_rel: float, pre_ms: float, post_ms: float,
                 max_lag_ms: float) -> dict:
    """ITD of the direct sound by cross-correlation (PLAN.md, "Direction check").

    Window: the same samples of both ears, from (onset - pre_ms) to (onset + post_ms), where onset is the first
    sample at which either ear reaches onset_rel x the larger ear peak (the direct sound arrives first at the
    nearer ear). Cross-correlation c(l) = sum_n L[n] R[n + l] for integer lags |l| <= max_lag_ms; the ITD is the
    lag of the largest (signed, not absolute) value of c, in ms; positive = right ear later = left ear first.
    A tie keeps the lag nearest zero. Also returns the level difference over the window (dB, left over right).
    """
    ir = np.asarray(ir, dtype=np.float64)
    if ir.ndim != 2 or ir.shape[0] < 2:
        raise ValueError(f"need a two-ear impulse response, got shape {ir.shape}")
    out = {"onset_index": None, "itd_ms": None, "lag_samples": None, "ild_db": None, "cue_side": NONE}
    t0 = common_onset_index(ir, onset_rel)
    if t0 is None:
        return out
    a = max(0, t0 - int(round(pre_ms * sr / 1000.0)))
    b = min(ir.shape[1], t0 + int(round(post_ms * sr / 1000.0)) + 1)
    L, R = ir[0, a:b], ir[1, a:b]
    max_lag = int(round(max_lag_ms * sr / 1000.0))
    n = len(L)
    best_lag, best = 0, -np.inf
    for lag in sorted(range(-max_lag, max_lag + 1), key=abs):
        if lag >= 0:
            c = float(np.dot(L[: n - lag], R[lag:])) if lag < n else 0.0
        else:
            c = float(np.dot(L[-lag:], R[: n + lag])) if -lag < n else 0.0
        if c > best:
            best, best_lag = c, lag
    el, er = float(np.sum(L * L)), float(np.sum(R * R))
    out.update({
        "onset_index": int(t0), "window": [int(a), int(b)], "lag_samples": int(best_lag),
        "itd_ms": 1000.0 * best_lag / sr,
        "ild_db": 10.0 * math.log10(el / er) if el > 0 and er > 0 else None,
        "cue_side": LEFT if best_lag > 0 else (RIGHT if best_lag < 0 else NONE),
    })
    return out


# ----------------------------------------------------------------------------- signals
def downmix(two_ear: np.ndarray) -> np.ndarray:
    """One-channel signal = mean of left and right of the same render (PLAN.md). Shape (2, n) -> (n,)."""
    x = np.asarray(two_ear)
    if x.ndim != 2 or x.shape[0] != 2:
        raise ValueError(f"expected shape (2, n), got {x.shape}")
    return 0.5 * (x[0] + x[1])


def looped(source: np.ndarray, start: int, n: int) -> np.ndarray:
    """n samples of the endlessly looped source, starting at sample index start (any integer, negative allowed:
    the sound has been playing before the episode starts)."""
    s = np.asarray(source)
    if s.ndim != 1 or len(s) == 0:
        raise ValueError("source must be a non-empty 1-D array")
    idx = (np.arange(n, dtype=np.int64) + int(start)) % len(s)
    return s[idx]


def to_float_mono(data: np.ndarray) -> np.ndarray:
    """WAV samples (any integer or float type, mono or multi-channel as (n, ch)) to float64 mono in [-1, 1]
    (multi-channel: mean of the channels, the way librosa.load(mono=True) folds channels)."""
    x = np.asarray(data)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min == 0:  # unsigned 8-bit WAV
            x = (x.astype(np.float64) - (info.max + 1) / 2.0) / ((info.max + 1) / 2.0)
        else:
            x = x.astype(np.float64) / float(-info.min)
    else:
        x = x.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x


def rms_dbfs(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))
    return 20.0 * math.log10(r) if r > 0 else -math.inf


def normalise_rms(x: np.ndarray, target_dbfs: float) -> tuple:
    """Scale so the RMS over the whole signal is target_dbfs (re full scale 1.0). Returns (signal, gain)."""
    cur = rms_dbfs(x)
    if not np.isfinite(cur):
        raise ValueError("silent source")
    g = 10.0 ** ((float(target_dbfs) - cur) / 20.0)
    return np.asarray(x, dtype=np.float64) * g, g


def _valid_conv(segment: np.ndarray, h: np.ndarray, n_out: int) -> np.ndarray:
    """Last n_out samples of the full convolution of segment with h, where segment holds len(h) - 1 samples of
    history followed by n_out new samples (so the result is the 'valid' part)."""
    m = len(segment) + len(h) - 1
    nfft = 1 << max(1, (m - 1).bit_length())
    y = np.fft.irfft(np.fft.rfft(segment, nfft) * np.fft.rfft(h, nfft), nfft)[:m]
    return y[len(h) - 1: len(h) - 1 + n_out]


def step_block(source: np.ndarray, ir: np.ndarray, block_start: int, n: int) -> np.ndarray:
    """Output samples [block_start, block_start + n) of the looped source played through the multi-channel
    impulse response ir (shape (ch, L)), as if the source had been playing forever with this response."""
    ir = np.atleast_2d(np.asarray(ir, dtype=np.float64))
    L = ir.shape[1]
    seg = looped(source, block_start - (L - 1), L - 1 + n)
    return np.stack([_valid_conv(seg, ir[c], n) for c in range(ir.shape[0])])


def synthesize(source: np.ndarray, irs: Sequence[np.ndarray], step_samples: int, start_offset: int,
               crossfade_samples: int) -> np.ndarray:
    """Listening clip: step k plays samples [k N, (k+1) N) of the looped source (N = step_samples, the video frame
    period), starting from source index start_offset, through that step's impulse response. As SoundSpaces 2.0's
    continuous simulator does (continuous_simulator.py crossfade, AUDIO.CROSSFADE), the first crossfade_samples of
    each step after the first fade linearly from the previous step's response to the current one (weights
    i / C and 1 - i / C for i = 0..C). Returns (ch, n_steps * N)."""
    N = int(step_samples)
    C = int(crossfade_samples)
    if C > N:
        raise ValueError("crossfade longer than a step")
    blocks = []
    prev = None
    for k, ir in enumerate(irs):
        start = int(start_offset) + k * N
        cur = step_block(source, ir, start, N)
        if prev is not None and C > 0:
            old = step_block(source, prev, start, N)
            w = np.arange(C + 1, dtype=np.float64) / C
            cur[:, : C + 1] = old[:, : C + 1] * (1.0 - w) + cur[:, : C + 1] * w
        blocks.append(cur)
        prev = ir
    if not blocks:
        return np.zeros((2, 0))
    return np.concatenate(blocks, axis=1)


def pad_irs(irs: Sequence[np.ndarray]) -> tuple:
    """Stack impulse responses of different lengths into (S, ch, Lmax) float32 plus their lengths."""
    lens = np.array([int(np.asarray(x).shape[1]) for x in irs], dtype=np.int64)
    ch = int(np.asarray(irs[0]).shape[0]) if len(irs) else 2
    out = np.zeros((len(irs), ch, int(lens.max()) if len(lens) else 0), dtype=np.float32)
    for i, x in enumerate(irs):
        out[i, :, : lens[i]] = x
    return out, lens


# ----------------------------------------------------------------------------- episode pick
FRAMES = ("identity", "mirror_x", "mirror_y", "mirror_xy")


def mirrored_scenes(lines: Iterable[str]) -> set:
    """Route A check 3 rule (summarize_check3.py scenes_summary, as rung 1 summarize.py): per scenes.jsonl record,
    totals per frame; a scene counts as possibly mirrored if a mirrored frame has the largest total (max() with
    identity first, so ties go to identity); records with all-zero totals are skipped."""
    out = set()
    for line in lines:
        if not line.strip():
            continue
        r = json.loads(line)
        d = r.get("frame_diag") or {}
        tot = {k: (d.get(k) or {}).get("total", 0) for k in FRAMES}
        if any(tot.values()) and max(tot, key=tot.get) != "identity":
            out.add(r["scene"])
    return out


def episode_id(scene: str, trajectory_id) -> str:
    """GN0's id, gn0/evaluation/common.py get_episode_id: '<scene>_<trajectory_id>'."""
    return f"{scene}_{trajectory_id}"


def scene_of(eid: str) -> str:
    return eid.rsplit("_", 1)[0]


def split_by_visibility(rows: Sequence[dict], mirrored: set, split: str, visible_col: str,
                        tol_col: str, tol: float) -> tuple:
    """Eligible ids in CSV row order, split into (visible, not visible). Eligible: status ok, scene not possibly
    mirrored. Every row must be of the split and carry the expected tolerance."""
    vis, nvis = [], []
    for i, r in enumerate(rows):
        if r["split"] != split:
            raise ValueError(f"row {i}: split {r['split']!r}, expected {split!r}")
        if abs(float(r[tol_col]) - float(tol)) > 1e-9:
            raise ValueError(f"row {i}: {tol_col} = {r[tol_col]}, expected {tol}")
        if r[visible_col] not in ("0", "1"):
            raise ValueError(f"row {i}: {visible_col} = {r[visible_col]!r}")
        if r["status"] != "ok" or r["scene"] in mirrored:
            continue
        (vis if r[visible_col] == "1" else nvis).append(episode_id(r["scene"], r["trajectory_id"]))
    if len(set(vis + nvis)) != len(vis) + len(nvis):
        raise ValueError("duplicate episode ids")
    return vis, nvis


def frozen_order(visible: Sequence[str], not_visible: Sequence[str], seed: int) -> List[str]:
    """Balanced frozen order: one numpy default_rng(seed) permutes the visible ids, then the not-visible ids; the
    two are interleaved visible, not visible, visible, ... until the smaller list runs out. Position 2k (0-based)
    is a visible start, 2k + 1 a not-visible start, so any even-length prefix is balanced."""
    rng = np.random.default_rng(int(seed))
    pv = rng.permutation(len(visible))
    pn = rng.permutation(len(not_visible))
    out = []
    for a, b in zip(pv, pn):
        out.append(visible[int(a)])
        out.append(not_visible[int(b)])
    return out


def order_text(order: Sequence[str]) -> str:
    return "".join(f"{x}\n" for x in order)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ----------------------------------------------------------------------------- tallies and the bar
def tally(steps: Iterable[dict]) -> dict:
    """Counts over per-step records (keys: counted, correct, los, cue_side). Occluded steps (no line of sight) are
    reported with their own sign agreement and no bar."""
    t = {"n_steps": 0, "n_los": 0, "n_occluded": 0, "n_counted": 0, "n_correct": 0,
         "n_occluded_with_side": 0, "n_occluded_correct": 0, "n_counted_no_cue": 0}
    for s in steps:
        t["n_steps"] += 1
        if s["los"]:
            t["n_los"] += 1
        else:
            t["n_occluded"] += 1
            if s["true_side"] != NONE:
                t["n_occluded_with_side"] += 1
                t["n_occluded_correct"] += int(s["cue_side"] == s["true_side"])
        if s["counted"]:
            t["n_counted"] += 1
            t["n_correct"] += int(bool(s["correct"]))
            t["n_counted_no_cue"] += int(s["cue_side"] == NONE)
    t["share_correct"] = t["n_correct"] / t["n_counted"] if t["n_counted"] else None
    t["occluded_share_correct"] = (t["n_occluded_correct"] / t["n_occluded_with_side"]
                                   if t["n_occluded_with_side"] else None)
    return t


def verdict(pooled: dict, bar: float, min_counted: int) -> str:
    """PASS if the pooled share of counted steps with the correct sign is at least bar; FAIL if below; TOO_FEW if
    fewer than min_counted steps were counted (then the batch says nothing and goes back to the lead)."""
    if pooled["n_counted"] < int(min_counted):
        return "TOO_FEW"
    return "PASS" if pooled["n_correct"] >= bar * pooled["n_counted"] - 1e-12 else "FAIL"


def wilson(k: int, n: int, z: float = 1.959963984540054) -> Optional[list]:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]
