"""Unit tests for summarize.py. The result rows below are hand-written to test the arithmetic only;
they are not results and nothing here writes to results/."""
import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
import yaml

import summarize as S

CFG = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
Z = CFG["pass_rule"]["wilson_z"]


def test_wilson_known_value():
    lo, hi = S.wilson(50, 100, Z)
    assert lo == pytest.approx(0.40383, abs=1e-5)
    assert hi == pytest.approx(0.59617, abs=1e-5)


def test_wilson_edges():
    lo, hi = S.wilson(0, 10, Z)
    assert lo == 0.0 and hi == pytest.approx(0.27753, abs=1e-5)
    lo, hi = S.wilson(10, 10, Z)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(0.72247, abs=1e-5)
    with pytest.raises(S.SummaryError):
        S.wilson(1, 0, Z)
    with pytest.raises(S.SummaryError):
        S.wilson(3, 2, Z)


def test_sanitize_like_gn0():
    assert S.sanitize(float("nan")) == 0.0
    assert S.sanitize(float("inf")) == 0.0
    assert S.sanitize("x") == 0.0
    assert S.sanitize(None) == 0.0
    assert S.sanitize("2.5") == 2.5


def _scene(name, identity, mx=0, my=0, mxy=0):
    return json.dumps({"scene": name, "frame_diag": {
        "identity": {"total": identity}, "mirror_x": {"total": mx},
        "mirror_y": {"total": my}, "mirror_xy": {"total": mxy}}})


def test_mirrored_rule(tmp_path):
    p = tmp_path / "scenes.jsonl"
    p.write_text("\n".join([
        _scene("A", 100, 10, 5, 1),     # identity wins
        _scene("B", 100, 101),          # mirror_x wins
        _scene("C", 50, 50),            # tie goes to identity
        _scene("D", 0, 0, 0, 0),        # all zero: skipped
        _scene("E", 3, 0, 0, 4),        # mirror_xy wins
        json.dumps({"scene": "F"}),     # no diagnostic: skipped
    ]) + "\n")
    assert S.mirrored_scenes(p) == {"B", "E"}


def _row(i, success, spl, osr, ne, tl, stop="model_stop"):
    return {"id": i, "status": "complete", "episode_complete": True, "success": success, "spl": spl,
            "oracle_success": osr, "distance_to_goal": ne, "path_length": tl, "stop": stop}


def _make_run(tmp_path, rows, ids_file=None, events=None):
    run = tmp_path / "run"
    (run / "gn0" / "log").mkdir(parents=True)
    for r in rows:
        (run / "gn0" / "log" / f"{r['id']}.json").write_text(json.dumps(r))
    if ids_file is not None:
        (run / "episode_ids.txt").write_text("".join(f"{i}\n" for i in ids_file))
    if events is not None:
        (run / "rung1_episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events) + '{"id": "half')
    return run


ORDER = ["s1_1", "s2_2", "s1_3", "s3_4", "s2_5", "s3_6"]
CHECK3 = {
    "s1_1": {"scene": "s1", "visible": True},
    "s2_2": {"scene": "s2", "visible": False},
    "s1_3": {"scene": "s1", "visible": False},
    "s3_4": {"scene": "s3", "visible": True},
    "s2_5": {"scene": "s2", "visible": True},
    "s3_6": {"scene": "s3", "visible": False},
}
ROWS = [
    _row("s1_1", 1.0, 0.8, 1.0, 1.0, 10.0),
    _row("s2_2", 0.0, 0.0, 1.0, 5.0, 12.0, stop="distance_to_goal_unchanged"),
    _row("s1_3", 1.0, 0.6, 1.0, 2.0, 8.0),
    _row("s3_4", 0.0, float("nan"), 0.0, 7.0, 6.0),
]
EVENTS = [
    {"id": "s1_1", "wall_s": 100.0, "first_in_launch": True},
    {"id": "s2_2", "wall_s": 20.0, "first_in_launch": False},
    {"id": "s1_3", "wall_s": 30.0, "first_in_launch": False},
    {"id": "s3_4", "wall_s": 50.0, "first_in_launch": False},
]


def _cfg_with_target(sr):
    cfg = copy.deepcopy(CFG)
    cfg["target"]["sr"] = sr
    return cfg


def test_summarize_groups_and_verdict(tmp_path):
    run = _make_run(tmp_path, ROWS, ids_file=ORDER[:4], events=EVENTS)
    out = S.summarize(_cfg_with_target(53.5), run, ORDER, 4, CHECK3, {"s2"}, usd_per_hour=3.6)
    g = out["groups"]["all"]
    assert g["n"] == 4 and g["successes"] == 2
    assert g["sr"] == pytest.approx(50.0)
    assert g["spl"] == pytest.approx(100 * (0.8 + 0.0 + 0.6 + 0.0) / 4)   # NaN read as 0, as GN0
    assert g["os"] == pytest.approx(75.0)
    assert g["ne"] == pytest.approx(15.0 / 4)
    assert g["tl"] == pytest.approx(36.0 / 4)
    lo, hi = S.wilson(2, 4, Z)
    assert g["sr_ci95"] == pytest.approx([100 * lo, 100 * hi])
    assert out["verdict"] == "reproduced"         # 53.5 inside a wide interval
    assert out["complete"] is True
    v = out["groups"]["visible"]
    assert v["n"] == 2 and v["successes"] == 1
    nv = out["groups"]["not_visible"]
    assert nv["n"] == 2 and nv["successes"] == 1
    assert out["groups"]["all_excl_mirrored"]["n"] == 3            # s2_2 dropped
    assert out["groups"]["not_visible_excl_mirrored"]["n"] == 1
    assert out["groups"]["visible_excl_mirrored"]["n"] == 2
    assert out["stop_reasons"] == {"model_stop": 3, "distance_to_goal_unchanged": 1}
    t = out["timing"]
    assert t["n_with_time"] == 4 and t["n_first_in_launch"] == 1
    assert t["total_s"] == pytest.approx(200.0)
    assert t["mean_s"] == pytest.approx(50.0)
    assert t["median_s"] == pytest.approx(40.0)
    assert t["mean_s_excl_first"] == pytest.approx(100.0 / 3)
    assert t["usd_per_episode"] == pytest.approx(50.0 / 3600 * 3.6)
    assert t["usd_per_episode_excl_first"] == pytest.approx(100.0 / 3 / 3600 * 3.6)


def test_verdict_not_reproduced_and_bounds(tmp_path):
    run = _make_run(tmp_path, ROWS)
    lo, hi = S.wilson(2, 4, Z)
    assert S.summarize(_cfg_with_target(95.0), run, ORDER, 4, CHECK3, set(), None)["verdict"] == "not reproduced"
    # the interval is closed: a target exactly on a bound counts as inside
    assert S.summarize(_cfg_with_target(100 * hi), run, ORDER, 4, CHECK3, set(), None)["verdict"] == "reproduced"
    assert S.summarize(_cfg_with_target(100 * lo), run, ORDER, 4, CHECK3, set(), None)["verdict"] == "reproduced"


def test_incomplete_run_has_no_verdict(tmp_path):
    run = _make_run(tmp_path, ROWS[:3])
    out = S.summarize(CFG, run, ORDER, 4, CHECK3, set(), None)
    assert out["verdict"] == "incomplete" and out["n_finished"] == 3
    assert out["timing"]["n_with_time"] == 0 and "usd_per_episode" not in out["timing"]


def test_extra_episode_fails(tmp_path):
    run = _make_run(tmp_path, ROWS + [_row("s3_6", 1.0, 1.0, 1.0, 0.5, 4.0)])
    with pytest.raises(S.SummaryError, match="not in the first 4"):
        S.summarize(CFG, run, ORDER, 4, CHECK3, set(), None)


def test_ids_file_must_be_prefix_of_order(tmp_path):
    run = _make_run(tmp_path, ROWS, ids_file=["s2_2", "s1_1", "s1_3", "s3_4"])
    with pytest.raises(S.SummaryError, match="not the first 4"):
        S.summarize(CFG, run, ORDER, 4, CHECK3, set(), None)


def test_non_binary_success_fails(tmp_path):
    run = _make_run(tmp_path, [_row("s1_1", 0.5, 0.1, 1.0, 1.0, 1.0)])
    with pytest.raises(S.SummaryError, match="0 or 1"):
        S.summarize(CFG, run, ORDER, 1, CHECK3, set(), None)


def test_incomplete_log_is_ignored(tmp_path):
    bad = dict(ROWS[0], status="running", episode_complete=False)
    run = _make_run(tmp_path, [bad])
    assert S.summarize(CFG, run, ORDER, 1, CHECK3, set(), None)["n_finished"] == 0


def test_load_check3_checks_tolerance(tmp_path):
    p = tmp_path / "c3.csv"
    p.write_text("row_index,scene,trajectory_id,status,visible_primary,depth_tol_primary_m\n"
                 "0,s1,1,ok,1,0.15\n1,s2,2,ok,0,0.15\n")
    got = S.load_check3(p, "visible_primary", "depth_tol_primary_m", 0.15)
    assert got == {"s1_1": {"scene": "s1", "visible": True}, "s2_2": {"scene": "s2", "visible": False}}
    with pytest.raises(S.SummaryError):
        S.load_check3(p, "visible_primary", "depth_tol_primary_m", 0.30)


def test_main_end_to_end(tmp_path):
    order = tmp_path / "order.txt"
    order.write_text("".join(f"{i}\n" for i in ORDER))
    c3 = tmp_path / "c3.csv"
    lines = ["row_index,scene,trajectory_id,status,visible_primary,depth_tol_primary_m"]
    for k, (i, v) in enumerate(CHECK3.items()):
        scene, traj = i.split("_")
        lines.append(f"{k},{scene},{traj},ok,{int(v['visible'])},0.15")
    c3.write_text("\n".join(lines) + "\n")
    sj = tmp_path / "scenes.jsonl"
    sj.write_text("\n".join([_scene("s1", 10), _scene("s2", 1, 9), _scene("s3", 10)]) + "\n")
    cfg = copy.deepcopy(CFG)
    cfg["episodes"]["order_sha256"] = hashlib.sha256(order.read_bytes()).hexdigest()
    cfg["episodes"]["source_sha256"] = hashlib.sha256(c3.read_bytes()).hexdigest()
    cfg["breakdowns"]["mirrored_scenes"]["source_sha256"] = hashlib.sha256(sj.read_bytes()).hexdigest()
    cfg["breakdowns"]["mirrored_scenes"]["n_scenes_expected"] = 1
    cfg["breakdowns"]["mirrored_scenes"]["n_test_unseen_episodes_expected"] = 2
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    run = _make_run(tmp_path, ROWS, ids_file=ORDER[:4], events=EVENTS)
    out = tmp_path / "summary.json"
    rc = S.main(["--config", str(cfg_path), "--run-dir", str(run), "--order-file", str(order), "--n", "4",
                 "--check3-csv", str(c3), "--scenes-jsonl", str(sj), "--usd-per-hour", "2.0", "--out", str(out)])
    assert rc == 0
    d = json.loads(out.read_text())
    assert d["groups"]["all_excl_mirrored"]["n"] == 3 and d["n_mirrored_scenes"] == 1
    # a wrong sha256 in the config stops the summary
    cfg["episodes"]["order_sha256"] = "0" * 64
    cfg_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(S.SummaryError, match="sha256"):
        S.main(["--config", str(cfg_path), "--run-dir", str(run), "--order-file", str(order), "--n", "4",
                "--check3-csv", str(c3), "--scenes-jsonl", str(sj), "--out", str(out)])
