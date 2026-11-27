"""Tests of first_frame_check.py's pure math on small synthetic arrays (no torch, no simulator, no cluster data).

Covers the plan's Design 3 to 6, the reading rule and the amendment of 2026-11-25:
distance normalization; nearest-neighbour r with the leave-self-out reference; the pooled median m; R; the task-level
bootstrap (resampling by task, a task drawn twice counts twice, ref not recomputed); zones, borderline wording and the
DINOv3 channel 757 rule; the outcome case; the three flag tests with the constant-channel exception; the mean-square
ratio wording; the per-channel statistics; Spearman; the correlate step on fake records.

Run on the Mac: /opt/homebrew/bin/python3.11 -I world-model-stacking-encoders/src/test_first_frame_check.py (or pytest)
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import first_frame_check as F  # noqa: E402

C, H, W = 3, 14, 28


def const(v, c=C):
    """A (c, 14, 28) array with every value v: distances between such arrays are |v1 - v2| after normalization."""
    return np.full((c, H, W), float(v))


def stack(vals, c=C):
    return np.stack([const(v, c) for v in vals])


def test_distance_normalization():
    a, b = const(0.0), const(2.5)
    assert abs(F.norm_dist(a, b) - 2.5) < 1e-12            # sqrt(P * 2.5^2) / sqrt(P)
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(C, H, W)), rng.normal(size=(C, H, W))
    assert abs(F.norm_dist(x, y) - np.linalg.norm(x - y) / np.sqrt(C * H * W)) < 1e-12
    # the normalization uses each group's own size: a 48-channel group and a 3-channel one read alike
    assert abs(F.norm_dist(const(0, 48), const(1, 48)) - 1.0) < 1e-12


def test_nearest_matches_brute_force_and_duplicates_are_zero():
    rng = np.random.default_rng(2)
    tr = rng.normal(size=(7, C, H, W))
    tr[5] = tr[2]                                             # an exact duplicate
    q = rng.normal(size=(4, C, H, W))
    d, idx = F.nearest_dists(q, tr)
    for i in range(4):
        bf = [F.norm_dist(q[i], tr[j]) for j in range(7)]
        assert abs(d[i] - min(bf)) < 1e-12 and abs(bf[idx[i]] - min(bf)) < 1e-12
    d_self, idx_self = F.nearest_dists(tr, tr, exclude_self=True)
    assert all(idx_self[i] != i for i in range(7))
    assert d_self[2] == 0.0 and d_self[5] == 0.0               # exactly 0, recomputed by direct difference
    for i in range(7):
        bf = min(F.norm_dist(tr[i], tr[j]) for j in range(7) if j != i)
        assert abs(d_self[i] - bf) < 1e-12


def test_task_ratios_leave_self_out_reference():
    # training values 0, 1, 3: nearest OTHER frame distances 1, 1, 2 -> ref = median = 1 (0 if self were kept)
    out = F.task_ratios(stack([0, 1, 3]), stack([0.5, 5, 3]))
    assert out["ref_dists"] == [1.0, 1.0, 2.0] and out["ref"] == 1.0
    assert np.allclose(out["d_live"], [0.5, 2.0, 0.0])        # the live frame is not left out of anything
    assert np.allclose(out["r"], [0.5, 2.0, 0.0])
    # ref is per task: a task with spacing 10 gives r 10 times smaller for the same live offsets
    out2 = F.task_ratios(stack([0, 10, 30]), stack([5, 50]))
    assert out2["ref"] == 10.0 and np.allclose(out2["r"], [0.5, 2.0])


def test_task_left_out_when_too_few_episodes_or_ref_zero():
    one = F.task_ratios(stack([1.0]), stack([0, 1]))
    assert one["n_train"] == 1 and one["r"] is None
    dup = F.task_ratios(stack([2, 2, 2, 7]), stack([3]))       # nn distances 0, 0, 0, 5 -> ref = 0
    assert dup["ref"] == 0.0 and dup["r"] is None


def _task(suite, ti, vals_by_group, live_by_group):
    ratios = {g: F.task_ratios(stack(vals_by_group[g]), stack(live_by_group[g])) for g in F.GROUPS}
    ratios[F.NO757] = F.task_ratios(stack(vals_by_group.get(F.NO757, vals_by_group[F.DINO])),
                                    stack(live_by_group.get(F.NO757, live_by_group[F.DINO])))
    return {"suite": suite, "task_index": ti, "name": f"t{ti}", "instruction": f"t {ti}",
            "init_states": list(range(len(live_by_group[F.VAE]))), "ratios": ratios}


def test_pooled_median_counts_each_frame_once():
    assert F.pooled_median([[1, 2, 3], [10, 10]]) == 3.0      # median of medians would be 6
    assert F.pooled_median([[4.0]]) == 4.0
    assert F.ratio_R(3.0, 2.0) == 1.5


def test_assemble_m_R_and_left_out_task_named():
    tr = {g: [0, 1, 3] for g in F.GROUPS}                     # ref 1 in every group
    lv = {F.VAE: [1.5, 3.5], F.DINO: [3, 6], "c-radiov4": [1.5, 3.5], "siglip2": [1.2, 4.0]}
    t0 = _task("libero_goal", 0, tr, lv)
    bad = _task("libero_goal", 1, {**tr, "siglip2": [5, 5, 5]}, lv)  # ref 0 in one group -> left out everywhere
    st = F.assemble([t0, bad])
    assert st["n_tasks"] == 1 and st["n_live_frames"] == 2
    assert st["left_out_tasks"] == [{"suite": "libero_goal", "task_index": 1, "name": "t1", "n_train": 3,
                                     "groups": ["siglip2"]}]
    # VAE r = [0.5, 0.5] -> m 0.5; DINOv3 r = [0, 3] -> m 1.5; R = 3
    assert st["m"][F.VAE] == 0.5 and st["m"][F.DINO] == 1.5 and st["R"][F.DINO] == 3.0
    assert st["R"]["c-radiov4"] == 1.0
    assert st["r_quantiles"][F.VAE]["all"]["median"] == 0.5
    assert "libero_goal" in st["r_quantiles"][F.VAE] and "libero_10" not in st["r_quantiles"][F.VAE]


def _manual_boot(per_task, encs, n_boot, seed):
    rng = np.random.default_rng(seed)
    K = len(per_task[F.VAE])
    out = {e: [] for e in encs}
    for _ in range(n_boot):
        idx = rng.integers(0, K, K)
        mv = np.median(np.concatenate([per_task[F.VAE][k] for k in idx]))
        for e in encs:
            out[e].append(np.median(np.concatenate([per_task[e][k] for k in idx])) / mv)
    return out


def test_bootstrap_resamples_tasks_with_ref_fixed():
    rng = np.random.default_rng(3)
    per_task = {g: [rng.uniform(0.5, 3, size=5) for _ in range(6)] for g in (F.VAE, "a", "b")}
    got = F.bootstrap(per_task, ["a", "b"], n_boot=200, seed=0)
    want = _manual_boot(per_task, ["a", "b"], 200, 0)
    for e in ("a", "b"):
        assert np.allclose(got[e]["draws"], want[e])
        assert np.allclose(got[e]["interval"], np.percentile(want[e], [2.5, 97.5]))
    # a task drawn twice counts twice: with 2 tasks, a draw [0, 0] has the median of task 0's frames twice over
    two = {F.VAE: [np.array([1.0, 1.0]), np.array([2.0, 2.0])], "a": [np.array([1.0, 1.0]), np.array([8.0, 8.0])]}
    r = np.random.default_rng(0)
    draws = F.bootstrap(two, ["a"], n_boot=50, seed=0)["a"]["draws"]
    for b in range(50):
        idx = r.integers(0, 2, 2)
        exp = np.median(np.concatenate([two["a"][k] for k in idx])) / np.median(np.concatenate([two[F.VAE][k] for k in idx]))
        assert abs(draws[b] - exp) < 1e-12
    # the r values are used as given (ref not recomputed): one task's r scaled by 2 changes the draws as the manual one
    scaled = {g: [x.copy() for x in v] for g, v in per_task.items()}
    scaled["a"][2] *= 2
    assert np.allclose(F.bootstrap(scaled, ["a"], n_boot=100, seed=0)["a"]["draws"],
                       _manual_boot(scaled, ["a"], 100, 0)["a"])
    # same seed, same draws; one task only: every draw is the point estimate
    assert F.bootstrap(per_task, ["a"], n_boot=10, seed=0)["a"]["draws"] == got["a"]["draws"][:10]
    one = {F.VAE: [np.array([1.0, 3.0])], "a": [np.array([2.0, 4.0])]}
    assert set(F.bootstrap(one, ["a"], n_boot=20)["a"]["draws"]) == {1.5}


def test_zones_and_borderline_wording():
    assert F.zone(0.7) == F.NEAR and F.zone(1.2) == F.NEAR
    assert F.zone(1.2000001) == F.UNCLEAR and F.zone(1.4999) == F.UNCLEAR
    assert F.zone(1.5) == F.CLEAR and F.zone(3) == F.CLEAR
    assert F.crossed_edges(1.0, 1.2) == [] and F.crossed_edges(1.25, 1.45) == []
    assert F.crossed_edges(1.1, 1.3) == [1.2] and F.crossed_edges(1.2, 1.4) == [1.2]
    assert F.crossed_edges(1.3, 1.6) == [1.5] and F.crossed_edges(1.3, 1.5) == [1.5]
    assert F.crossed_edges(1.1, 1.7) == [1.2, 1.5]
    rd = F.read_encoder(1.3, [1.1, 1.6])
    assert rd["zone"] == F.UNCLEAR and rd["reading"] == F.UNCLEAR and rd["borderline"]
    assert rd["wording"] == "unclear, borderline: the bootstrap interval crosses 1.2 and 1.5"
    # the zone and the reading come from the point estimate; a borderline interval changes neither
    rd = F.read_encoder(1.1, [0.9, 1.25])
    assert rd["zone"] == F.NEAR and rd["reading"] == F.NEAR and rd["wording"] == "near, borderline: the bootstrap interval crosses 1.2"
    rd = F.read_encoder(1.6, [1.55, 1.9])
    assert rd["reading"] == F.CLEAR and not rd["borderline"] and rd["wording"] == "clearly larger"


def test_channel_757_rule():
    same = F.read_encoder(1.6, [1.5, 1.8], R_no757=1.7)
    assert same["reading"] == F.CLEAR and not same["depends_on_757"]
    diff = F.read_encoder(1.6, [1.5, 1.8], R_no757=1.1)        # zone uses all channels; differs without 757
    assert diff["zone"] == F.CLEAR and diff["zone_without_757"] == F.NEAR
    assert diff["reading"] == F.UNCLEAR and diff["depends_on_757"] and "depends on that channel" in diff["wording"]
    near = F.read_encoder(1.1, [1.0, 1.15], R_no757=1.3)
    assert near["reading"] == F.UNCLEAR
    # channel 757 is removed along the channel axis only
    x = np.arange(2 * 1024 * 2).reshape(2, 1024, 2)
    y = F.drop_channel(x, 757)
    assert y.shape == (2, 1023, 2) and (y[:, 756] == x[:, 756]).all() and (y[:, 757] == x[:, 758]).all()


def test_outcome_cases_in_order():
    assert F.outcome_case({"dinov3": F.NEAR, "c-radiov4": F.CLEAR, "siglip2": F.UNCLEAR})["case"] == 1
    assert F.outcome_case({"dinov3": F.NEAR, "c-radiov4": F.NEAR, "siglip2": F.NEAR})["case"] == 2
    assert F.outcome_case({"dinov3": F.NEAR, "c-radiov4": F.UNCLEAR, "siglip2": F.NEAR})["case"] == 3
    o = F.outcome_case({"dinov3": F.CLEAR, "c-radiov4": F.CLEAR, "siglip2": F.NEAR})
    assert o["case"] == 1 and o["clearly_larger"] == ["dinov3", "c-radiov4"]


def test_flag_tests_and_constant_exception():
    #            ok    a:sd lo  a:sd hi  a:mean  b:sd_L lo  b:sd_L hi  c:shift  constant  ok edge
    mean_T = np.array([0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 2.0])
    sd_T = np.array([1.0, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5e-7, 4.0])
    mean_L = np.array([0.5, 0.0, 0.0, 2.5, 0.0, 0.0, 2.1, 50.0, 2.0 + 8.0])
    sd_L = np.array([1.5, 0.2, 5.0, 1.0, 0.2, 4.5, 1.0, 9.0, 16.0])
    fl = F.channel_flags(mean_T, sd_T, mean_L, sd_L)
    assert fl["a"] == [1, 2, 3, 7]                           # channel 7 (sd_T 5e-7 < 0.25) is also an (a) flag
    assert fl["b"] == [4, 5]                                  # sd_L 4 x sd_T exactly (channel 8) is not flagged
    assert fl["c"] == [6]                                     # shift exactly 2 sd_T (channel 8) is not flagged
    assert fl["constant_T"] == [7]                            # (b) and (c) not applied to it
    rep = F.flag_report(fl, 9)
    assert rep["a"]["count"] == 4 and abs(rep["a"]["share"] - 4 / 9) < 1e-12


def test_mean_square_ratio_wording():
    for r, very in ((4.01, True), (0.249, True), (4.0, False), (0.25, False), (1.0, False), (2.0, False)):
        ratio, text = F.range_wording(r * 3.0, 3.0)
        assert abs(ratio - r) < 1e-12
        assert text.startswith("in very different ranges" if very else "the overall ranges are comparable"), (r, text)


def test_channel_stats_against_numpy():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(9, 4, H, W))
    x[:, 0] = 1e4 + 1e-2 * x[:, 0]                            # large offset, tiny spread: shifted sums stay accurate
    x[:, 3] *= 7
    st = F.ChannelStats(4)
    for chunk in (x[:2], x[2:3], x[3:]):                      # added task by task
        st.add(chunk)
    v = np.moveaxis(x, 1, 0).reshape(4, -1)
    assert st.n == 9 * H * W
    assert np.allclose(st.mean(), v.mean(1), rtol=0, atol=1e-9)
    assert np.allclose(st.sd(), v.std(1, ddof=0), rtol=1e-9)  # population sd (divide by N)
    assert np.allclose(st.maxabs, np.abs(v).max(1))
    assert abs(st.mean_square() - (v ** 2).mean()) < 1e-6 * (v ** 2).mean()
    s = F.group_summary(st)
    assert s["sd_min"] == st.sd().min() and s["max_abs"] == np.abs(v).max()


def test_ranges_report_end_to_end():
    rng = np.random.default_rng(5)
    stats = {"T": {}, "L": {}}
    for g, c, scale in ((F.VAE, 2, 1.0), ("dinov3", 3, 1.0), ("c-radiov4", 3, 3.0), ("siglip2", 3, 1.0)):
        for s, shift in (("T", 0.0), ("L", 0.0)):
            stats[s][g] = F.ChannelStats(c)
            stats[s][g].add(scale * rng.normal(size=(20, c, H, W)) + shift)
    rep = F.ranges_report(stats)
    assert rep["wording"]["c-radiov4"]["text"].startswith("in very different ranges")      # about 9 x
    assert rep["wording"]["dinov3"]["text"].startswith("the overall ranges are comparable")
    assert set(rep["per_channel"]) == set(F.GROUPS) and len(rep["per_channel"]["dinov3"]["T"]["sd"]) == 3
    assert rep["flags"]["c-radiov4"]["a"]["count"] == 0 and set(rep["mean_square_ratio_to_vae"]) == {"T", "L"}
    json.dumps(rep, allow_nan=False)


def test_spearman_with_ties_and_constant():
    x = [1, 2, 2, 3, 5, 4]
    y = [2, 1, 4, 3, 6, 5]
    rx = F.rankdata(x)
    assert list(rx) == [1, 2.5, 2.5, 4, 6, 5]
    try:
        from scipy.stats import spearmanr
        assert abs(F.spearman(x, y) - spearmanr(x, y).statistic) < 1e-12
    except ImportError:
        pass
    assert abs(F.spearman([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-12
    assert F.spearman([1, 1, 1], [1, 2, 3]) is None


def test_task_level_ratio_and_757_live_and_latent_diff():
    rt = F.task_level_ratios({F.VAE: [1, 2, 4], "dinov3": [3, 6, 9]}, ["dinov3"])
    assert rt == {"dinov3": 3.0}
    z = F.channel_757_live(np.array([[-3.0, 1.0], [2.0, 0.5]]))
    assert z["median"] == 0.75 and z["max_abs"] == 3.0
    ma, rl = F.latent_diff(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
    assert ma == 1.0 and abs(rl - 1 / np.sqrt(2)) < 1e-12
    assert F.latent_diff(np.zeros(2), np.zeros(2)) == (0.0, None)
    s = F.latent_diff_summary([{"suite": "libero_10", "max_abs_diff": 0.0, "rel_l2": 0.0},
                               {"suite": "libero_10", "max_abs_diff": 2.0, "rel_l2": None}])
    assert s["all"]["n_identical"] == 1 and s["all"]["n_rel_l2_undefined"] == 1 and s["libero_goal"] is None


def test_correlate_on_fake_records():
    with tempfile.TemporaryDirectory() as d:
        recs = {}
        rates = {"none": [1.0, 0.8, 0.6], "dinov3": [0.9, 0.4, 0.6], "c-radiov4": [1.0, 0.8, 0.0],
                 "siglip2": [0.5, 0.8, 0.6]}
        for c, rr in rates.items():
            files = []
            for sh in range(4):
                p = Path(d) / f"{c}-{sh}.jsonl"
                lines = []
                for ti, rate in enumerate(rr):
                    for trial in range(sh, 10, 4):            # 10 trials per task over 4 shards
                        lines.append(json.dumps({"suite": "libero_goal", "episode_id": ti * 1000 + trial,
                                                 "eval_seed": 41, "success": int(trial < rate * 10),
                                                 "instruction": f"t {ti}"}))
                p.write_text("\n".join(lines) + "\n")
                files.append(p)
            recs[c] = files
        result = {"statistics": {"per_task_ratio": [
            {"suite": "libero_goal", "task_index": ti, "name": f"t{ti}", "instruction": f"t {ti}",
             "dinov3": [2.0, 3.0, 1.0][ti], "c-radiov4": [1.0, 2.0, 3.0][ti], "siglip2": [3.0, 2.0, 1.0][ti],
             "dinov3_without_757": 1.0} for ti in range(3)]}}
        out = F.correlate(result, recs)
        assert abs(out["spearman"]["dinov3"]["spearman"] - 1.0) < 1e-12        # drops 0.1, 0.4, 0.0 vs 2, 3, 1
        # siglip2: drops 0.5, 0, 0 (a tie) against ratios 3, 2, 1: ranks 3, 1.5, 1.5 vs 3, 2, 1
        assert abs(out["spearman"]["siglip2"]["spearman"] - F.spearman([3, 2, 1], [0.5, 0, 0])) < 1e-12
        assert abs(F.spearman([3, 2, 1], [0.5, 0, 0]) - np.sqrt(3) / 2) < 1e-12
        assert out["rows"][2]["drop_c-radiov4"] == 0.6
        bad = json.loads(json.dumps(result))
        bad["statistics"]["per_task_ratio"][0]["instruction"] = "other"
        try:
            F.correlate(bad, recs)
            raise AssertionError("instruction mismatch not caught")
        except F.CheckFailed:
            pass


def test_no_torch_needed_for_the_pure_part():
    import importlib                                          # importing first_frame_check must not import torch
    import subprocess
    code = "import sys; sys.path.insert(0, %r); import first_frame_check; print('torch' in sys.modules)" % \
        str(Path(F.__file__).parent)
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False", out
    importlib.reload(F)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok {name}")
    print(f"{len(tests)} tests passed")
