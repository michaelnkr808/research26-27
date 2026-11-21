"""Restrict the released GN0 remote client to the rung 1 episode list and log per-episode wall time.

Two functions of gn0/evaluation/common.py (GN0 c927781) are wrapped, nothing else:

- select_eval_episodes: called once by gn0.evaluation.remote.run_exp (line 354) after the dataset is
  loaded and sorted. The wrapper calls the original (which drops episodes already finished in log/),
  then keeps only the requested ids, listed in the frozen order. GN-Bench then reshuffles the chosen
  episodes (test_unseen.yaml ITERATOR_OPTIONS SHUFFLE true), so they run in a random order; the set
  is what is fixed (PLAN.md, "Other choices"). It fails if a requested id is not in the
  dataset, if the start/end range is not the full dataset, or if log/ holds a finished episode that
  is not requested (results of another run).
- atomic_write_json: called by save_episode_metrics (remote.py line 607) once per finished episode.
  After the original write, the wrapper appends one JSON line (id, GN0 metrics, wall time) to
  RUNG1_EVENTS and fsyncs it. Wall time is the time since the previous episode was saved, or, for the
  first episode of a launch, since episode selection (so it includes the simulator and connection
  start-up; first_in_launch marks it).

Environment: RUNG1_EPISODE_IDS (file, one id per line), RUNG1_EVENTS (jsonl to append),
RUNG1_HOOK_STATUS (json written once the filter has run), RUNG1_LAUNCH_ID.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Iterable

METRIC_KEYS = ("distance_to_goal", "success", "spl", "path_length", "oracle_success")


class FilterError(RuntimeError):
    pass


def read_ids(path: str | Path) -> list[str]:
    ids = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise FilterError(f"no episode ids in {path}")
    if len(set(ids)) != len(ids):
        raise FilterError(f"duplicate ids in {path}")
    return ids


def select_in_order(all_episodes: list, pending: list, wanted: list[str], finished_ids: Iterable[str],
                    id_fn: Callable) -> tuple[list, dict]:
    """Return the pending episodes whose id is wanted, in the order of wanted, and a status dict."""
    all_ids = [id_fn(ep.ref_json) for ep in all_episodes]
    all_set = set(all_ids)
    if len(all_set) != len(all_ids):
        raise FilterError("the dataset holds duplicate episode ids")
    missing = [i for i in wanted if i not in all_set]
    if missing:
        raise FilterError(f"{len(missing)} requested ids are not in the dataset, e.g. {missing[:3]}")
    wanted_set = set(wanted)
    finished = set(finished_ids)
    foreign = sorted(finished - wanted_set)
    if foreign:
        raise FilterError(f"log/ holds {len(foreign)} finished episodes that are not requested, e.g. {foreign[:3]}")
    by_id = {id_fn(ep.ref_json): ep for ep in pending}
    selected = [by_id[i] for i in wanted if i in by_id]
    status = {
        "n_requested": len(wanted),
        "n_dataset": len(all_ids),
        "n_already_finished": len(finished & wanted_set),
        "n_selected_now": len(selected),
    }
    if status["n_already_finished"] + status["n_selected_now"] != len(wanted):
        raise FilterError(f"bookkeeping mismatch: {status}")
    return selected, status


def event_line(payload: dict, now: float, t_prev: float, first: bool, launch_id: str) -> dict:
    line = {"id": payload["id"], "t_end_unix": now, "wall_s": now - t_prev,
            "first_in_launch": first, "launch_id": launch_id, "stop": payload.get("stop")}
    for k in METRIC_KEYS:
        if k in payload:
            line[k] = payload[k]
    return line


def append_line(path: str | Path, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


class _State:
    log_dir: Path | None = None
    t_prev: float = 0.0
    first: bool = True


def install() -> None:
    import gn0.evaluation.common as common

    ids_path = os.environ["RUNG1_EPISODE_IDS"]
    events_path = os.environ["RUNG1_EVENTS"]
    status_path = os.environ["RUNG1_HOOK_STATUS"]
    launch_id = os.environ.get("RUNG1_LAUNCH_ID", "unknown")
    wanted = read_ids(ids_path)
    state = _State()

    orig_select = common.select_eval_episodes
    orig_write = common.atomic_write_json

    def select_eval_episodes(dataset_split, log_dir, start_idx, end_idx):
        if start_idx != 0 or end_idx not in (None, -1):
            raise FilterError(f"expected the full range, got start {start_idx} end {end_idx}")
        pending, resolved_end, _skipped, finished_count = orig_select(dataset_split, log_dir, start_idx, end_idx)
        finished_ids = common.load_finished_episode_ids(Path(log_dir))
        selected, status = select_in_order(dataset_split.episodes, pending, wanted, finished_ids,
                                           common.get_episode_id)
        state.log_dir = Path(log_dir).resolve()
        state.t_prev = time.time()
        state.first = True
        status.update({"launch_id": launch_id, "time_unix": state.t_prev, "ids_file": ids_path})
        tmp = Path(status_path).with_suffix(".tmp")
        tmp.write_text(json.dumps(status, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, status_path)
        print(f"[rung1] episode filter attached: {status}", flush=True)
        return selected, resolved_end, len(wanted) - len(selected), finished_count

    def atomic_write_json(path, payload):
        orig_write(path, payload)
        p = Path(path)
        if (state.log_dir is not None and p.parent.resolve() == state.log_dir
                and isinstance(payload, dict) and payload.get("status") == "complete"):
            now = time.time()
            append_line(events_path, event_line(payload, now, state.t_prev, state.first, launch_id))
            state.t_prev = now
            state.first = False

    common.select_eval_episodes = select_eval_episodes
    common.atomic_write_json = atomic_write_json
