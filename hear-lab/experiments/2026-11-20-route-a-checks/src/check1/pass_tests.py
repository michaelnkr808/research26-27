"""Check 1 pass tests (PLAN.md, check 1, "Pass (before running)"), as code that prints PASS/FAIL
with the numbers.

First arrival: the first sample whose magnitude reaches FIRST_ARRIVAL_REL_THRESHOLD times the channel's
peak magnitude. This threshold is not in the PLAN or config.yaml; it is this code's choice (see RUN.md,
"Choices the lead should confirm"). Onsets at ONSET_REPORT_THRESHOLDS are also written so that a
different threshold can be read off without rerunning.
Binaural channel order: channel 0 is taken as the left ear and channel 1 as the right ear. The header
does not state the order; a FAIL of the left-before-right test can therefore mean swapped channels.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

FIRST_ARRIVAL_REL_THRESHOLD = 0.1
ONSET_REPORT_THRESHOLDS = (0.01, 0.1, 0.5)
LEFT_CHANNEL, RIGHT_CHANNEL = 0, 1


def first_arrival_index(x: np.ndarray, rel: float = FIRST_ARRIVAL_REL_THRESHOLD) -> Optional[int]:
    a = np.abs(np.asarray(x, dtype=np.float64))
    if a.size == 0:
        return None
    m = float(a.max())
    if not np.isfinite(m) or m <= 0:
        return None
    return int(np.argmax(a >= rel * m))


def onsets_ms(x: np.ndarray, sr: float) -> Dict[str, Optional[float]]:
    out = {}
    for r in ONSET_REPORT_THRESHOLDS:
        i = first_arrival_index(x, r)
        out[f"rel_{r:g}"] = None if i is None else 1000.0 * i / sr
    return out


def _line(ok: bool, name: str, run: str, text: str) -> str:
    return f"{'PASS' if ok else 'FAIL'} {name} [{run}]: {text}"


def test_basic(ir: np.ndarray, expected_channels: int, name: str, run: str) -> Dict:
    finite = bool(np.all(np.isfinite(ir)))
    nonzero = bool(np.any(ir != 0))
    nch = int(ir.shape[0])
    ok = finite and nonzero and nch == expected_channels and ir.shape[1] > 0
    text = (f"channels {nch} (expected {expected_channels}), samples {ir.shape[1]}, finite {finite}, "
            f"non-zero {nonzero}, peak |x| {float(np.abs(ir).max()) if ir.size else 0.0:.6g}")
    return {"test": name, "run": run, "pass": ok, "line": _line(ok, name, run, text),
            "channels": nch, "samples": int(ir.shape[1]), "finite": finite, "nonzero": nonzero}


def test_first_arrival(ir_mono: np.ndarray, distance_m: float, sr: float, c: float, tol_ms: float,
                       run: str, los: bool) -> Dict:
    name = "first_arrival_vs_distance"
    i = first_arrival_index(ir_mono[0])
    expected = 1000.0 * distance_m / c
    if i is None or not los:
        why = "no line-of-sight listener placement found" if not los else "IR has no non-zero sample"
        return {"test": name, "run": run, "pass": False, "line": _line(False, name, run, why),
                "expected_ms": expected}
    measured = 1000.0 * i / sr
    diff = measured - expected
    ok = abs(diff) <= tol_ms
    text = (f"measured {measured:.3f} ms (sample {i}, threshold {FIRST_ARRIVAL_REL_THRESHOLD:g} x peak), "
            f"expected {expected:.3f} ms ({distance_m:.3f} m / {c:g} m/s), diff {diff:+.3f} ms, tolerance {tol_ms:g} ms")
    return {"test": name, "run": run, "pass": ok, "line": _line(ok, name, run, text),
            "measured_ms": measured, "expected_ms": expected, "diff_ms": diff,
            "distance_m": distance_m, "onsets_ms": onsets_ms(ir_mono[0], sr)}


def test_left_before_right(ir_bin: np.ndarray, sr: float, run: str, placed: bool) -> Dict:
    name = "left_ear_before_right_ear"
    if not placed or ir_bin.shape[0] < 2:
        why = "no open-floor placement found" if not placed else f"{ir_bin.shape[0]} channels"
        return {"test": name, "run": run, "pass": False, "line": _line(False, name, run, why)}
    il = first_arrival_index(ir_bin[LEFT_CHANNEL])
    ir_ = first_arrival_index(ir_bin[RIGHT_CHANNEL])
    if il is None or ir_ is None:
        return {"test": name, "run": run, "pass": False, "line": _line(False, name, run, "a channel is all zero")}
    ok = il < ir_
    text = (f"left (ch {LEFT_CHANNEL}) {1000.0 * il / sr:.3f} ms, right (ch {RIGHT_CHANNEL}) {1000.0 * ir_ / sr:.3f} ms, "
            f"left leads by {1000.0 * (ir_ - il) / sr:+.3f} ms ({ir_ - il} samples); "
            f"peak ratio left/right {float(np.abs(ir_bin[LEFT_CHANNEL]).max() / max(np.abs(ir_bin[RIGHT_CHANNEL]).max(), 1e-30)):.3f}")
    return {"test": name, "run": run, "pass": ok, "line": _line(ok, name, run, text),
            "left_ms": 1000.0 * il / sr, "right_ms": 1000.0 * ir_ / sr,
            "onsets_left_ms": onsets_ms(ir_bin[LEFT_CHANNEL], sr), "onsets_right_ms": onsets_ms(ir_bin[RIGHT_CHANNEL], sr)}
