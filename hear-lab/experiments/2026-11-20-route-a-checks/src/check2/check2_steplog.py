"""Check 2 timing and memory callback (PLAN.md, Amendment 2026-11-20 (b), step 5).

A Hugging Face TrainerCallback that, at on_train_begin, at every on_step_end (once per
optimizer step) and at on_train_end, calls torch.cuda.synchronize() and appends one JSON
line to <args.output_dir>/check2_steps.jsonl with the wall-clock time and torch's memory
counters. It never resets the peak counters, so the on_train_begin line holds the peak of
setup (model build, DeepSpeed init, optimizer state) and the last line the peak since
process start.

It is attached by sitecustomize.py in this folder; nothing in WNM-3D is changed.

Also a small summariser for the file it writes:
    python check2_steplog.py summarize <run>/check2_steps.jsonl --warmup 10 --batch 12
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

try:  # the hook imports this module only after transformers.trainer has been imported
    from transformers import TrainerCallback as _Base
except Exception:  # allows the summariser to run without transformers
    _Base = object

STEPS_FILENAME = "check2_steps.jsonl"
MEMORY_FIELDS = (
    "memory_allocated",
    "max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
)


def _default_cuda():
    import torch

    return torch.cuda


class Check2StepLogger(_Base):
    """Writes one JSON line per optimizer step. cuda and clock are injectable for tests."""

    def __init__(self, cuda=None, clock=time.time, output_path: str | None = None):
        self._cuda = cuda
        self._clock = clock
        self._output_path = output_path

    def _cuda_mod(self):
        if self._cuda is None:
            self._cuda = _default_cuda()
        return self._cuda

    def _path(self, args) -> str:
        if self._output_path is None:
            self._output_path = os.path.join(args.output_dir, STEPS_FILENAME)
        return self._output_path

    def record(self, args, state, event: str) -> dict:
        cuda = self._cuda_mod()
        if not cuda.is_available():
            raise RuntimeError("check2 step logger: CUDA is not available; check 2 needs the GPU")
        cuda.synchronize()  # GPU work queued so far is finished before the time is read
        rec = {
            "event": event,
            "global_step": int(getattr(state, "global_step", 0) or 0),
            "time": self._clock(),
        }
        rec["memory_allocated"] = int(cuda.memory_allocated())
        rec["max_memory_allocated"] = int(cuda.max_memory_allocated())
        rec["memory_reserved"] = int(cuda.memory_reserved())
        rec["max_memory_reserved"] = int(cuda.max_memory_reserved())
        rec["pid"] = os.getpid()
        rec["local_rank"] = os.environ.get("LOCAL_RANK")
        if getattr(state, "is_world_process_zero", True):
            path = self._path(args)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
        return rec

    def on_train_begin(self, args, state, control, **kwargs):
        self.record(args, state, "train_begin")

    def on_step_end(self, args, state, control, **kwargs):
        self.record(args, state, "step_end")

    def on_train_end(self, args, state, control, **kwargs):
        self.record(args, state, "train_end")


def summarize(records: list[dict], warmup: int, batch: int | None = None) -> dict:
    """Step times from consecutive step_end lines; peaks at train_begin and at the last line.

    Step k's time is t(step_end k) - t(step_end k-1); step 1 is measured from train_begin.
    Timed steps are those with global_step > warmup (the first timed interval runs from
    step warmup to step warmup + 1).
    """
    begin = [r for r in records if r["event"] == "train_begin"]
    steps = sorted((r for r in records if r["event"] == "step_end"), key=lambda r: r["global_step"])
    if not begin or not steps:
        raise ValueError("need a train_begin line and at least one step_end line")
    numbers = [r["global_step"] for r in steps]
    if numbers != list(range(1, len(steps) + 1)):
        raise ValueError(f"step numbers are not 1..N in order: {numbers[:20]}")
    prev_t = begin[0]["time"]
    durations = []
    for r in steps:
        durations.append(r["time"] - prev_t)
        prev_t = r["time"]
    timed = durations[warmup:]
    out = {
        "n_steps": len(steps),
        "warmup_steps": warmup,
        "timed_steps": len(timed),
        "step_seconds_all": durations,
    }
    if timed:
        med = statistics.median(timed)
        out.update(
            timed_median_s=med,
            timed_mean_s=statistics.fmean(timed),
            timed_min_s=min(timed),
            timed_max_s=max(timed),
            timed_total_s=sum(timed),
        )
        if batch:
            out["samples_per_s_at_median"] = batch / med
    last = records[-1]
    out["setup_peak"] = {k: begin[0][k] for k in MEMORY_FIELDS}
    out["final_peak"] = {k: last[k] for k in MEMORY_FIELDS}
    out["peak_allocated_from_setup"] = begin[0]["max_memory_allocated"] == last["max_memory_allocated"]
    out["peak_reserved_from_setup"] = begin[0]["max_memory_reserved"] == last["max_memory_reserved"]
    return out


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="check 2 step log tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("path")
    s.add_argument("--warmup", type=int, required=True)
    s.add_argument("--batch", type=int, default=None, help="per-device batch (no accumulation)")
    a = p.parse_args(argv)
    print(json.dumps(summarize(read_jsonl(a.path), a.warmup, a.batch), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
