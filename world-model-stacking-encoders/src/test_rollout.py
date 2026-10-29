"""Synthetic checks for rollout.py (bookkeeping, processing, batching; no simulator). Run: python test_rollout.py"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

import lp_eval as L
import proxy_wam as PW
import rollout as R
import t5_cache


# ---- fake simulator for run_jobs: deterministic in (episode, eval seed, actions) -------------------------------------

class FakeEnv:
    """Success once the summed x-action passes a target that depends on the episode; raises for episode 13."""

    def __init__(self, job):
        self.job, self.rng = job, None

    def seed(self, s):
        self.rng = np.random.default_rng(s)

    def reset(self):
        pass

    def set_init_state(self, init):
        self.x, self.target = 0.0, init
        return self.obs()

    def obs(self):
        img = np.full((R.RENDER, R.RENDER, 3), int(self.x * 10) % 256, np.uint8)
        return {"agentview_image": img, "robot0_eye_in_hand_image": img, "robot0_eef_pos": np.array([self.x, 0, 0]),
                "robot0_eef_quat": np.array([0, 0, 0, 1.0]), "robot0_gripper_qpos": np.array([0.04, -0.04])}

    def step(self, a):
        if self.job["episode_id"] == 13:
            raise ValueError("planted failure")
        self.x += float(a[0]) + 0.001 * float(self.rng.standard_normal())
        return self.obs(), 0.0, self.x >= self.target, {}

    def close(self):
        pass


def fake_env(job):
    target = 0.5 + (job["episode_id"] % 5) * 0.3 + (job["eval_seed"] - 41) * 0.2
    target += 100.0 * (job["episode_id"] % 2 == 0)  # even episodes can't reach it: they run to max steps and fail
    env = FakeEnv(job)
    env.seed(job["eval_seed"])                     # as rollout.libero_env
    return env, target, f"task {job['episode_id'] % 3}", ""


def fake_act(reqs):
    """Per-item deterministic policy: x step from the eval seed and the proprio x."""
    out = []
    for r in reqs:
        step = 0.01 * (1 + r["eval_seed"] % 3) * (1 + 0.1 * math.sin(r["state"][0]))
        out.append(np.tile([step, 0, 0, 0, 0, 0, 1.0], (PW.HORIZON, 1)).astype(np.float32))
    return out


def main():
    # 1. Instructions: perturbation suffixes stripped back to the base instruction; rewritten language kept;
    #    "from_table_center" is part of a base name, not a suffix.
    cases = [
        ("KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_table_1",
         "turn on the stove and put the moka pot on it table 1", "turn on the stove and put the moka pot on it"),
        ("LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_view_0_0_100_0_0_initstate_1",
         "put both the alphabet soup and the tomato sauce in the basket view 0 0 100 0 0 initstate 1",
         "put both the alphabet soup and the tomato sauce in the basket"),
        ("KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it_tb_3",
         "put the butter at the back in the top drawer of the cabinet and close it tb 3",
         "put the butter at the back in the top drawer of the cabinet and close it"),
        ("pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_table_11",
         "pick up the black bowl from table center and place it on the plate table 11",
         "pick up the black bowl from table center and place it on the plate"),
        ("open_the_middle_drawer_of_the_cabinet_moved_level1_sample1",
         "open the middle drawer of the cabinet moved level1 sample1", "open the middle drawer of the cabinet"),
        ("pick_up_the_black_bowl_on_the_plate_level1_sample4", "pick up the black bowl on the plate level1 sample4",
         "pick up the black bowl on the plate"),
        ("put_the_bowl_on_the_plate_add_14", "put the bowl on the plate add 14", "put the bowl on the plate"),
        ("turn_on_the_stove_light_1", "turn on the stove light 1", "turn on the stove"),
        ("turn_on_the_stove", "turn on the stove", "turn on the stove"),
        ("pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_language_1_view_0_0_100_0_0_initstate_0",
         "Could you grab the black bowl sitting between the plate and ramekin, then set it on the plate?",
         "Could you grab the black bowl sitting between the plate and ramekin, then set it on the plate?"),
    ]
    for name, lang, want in cases:
        assert R.lp_instruction(name, lang) == want, (name, R.lp_instruction(name, lang))
    try:
        R.lp_instruction("turn_on_the_stove_table_1", "put the bowl on the plate table 1")
        raise RuntimeError("mismatched base accepted")
    except AssertionError:
        pass
    print(f"1 ok: instructions ({len(cases)} name formats, mismatch rejected)")

    # 2. Jobs: every subset episode x eval seed 41-43, clean trials once, keys unique; sharding partitions them.
    sub = {"subset_id": "abc", "episodes": [{"suite": "libero_goal", "id": i, "name": f"t{i}", "category": "c"}
                                            for i in range(1, 6)]}
    jobs = R.make_jobs(sub, 2, {"libero_goal": ["a", "b"], "libero_10": ["c"]})
    lp = [j for j in jobs if j["benchmark"] == L.LP]
    assert len(lp) == 15 and {j["eval_seed"] for j in lp} == {41, 42, 43}
    assert len(jobs) == 15 + 3 * 2 and len({R.job_key(j) for j in jobs}) == len(jobs)
    shards = [jobs[i::4] for i in range(4)]
    assert sorted(map(R.job_key, sum(shards, []))) == sorted(map(R.job_key, jobs))
    print("2 ok: jobs (episodes x eval seeds, clean trials, shards partition)")

    # 3. Observation processing: 180 deg rotation of both cameras, 224 crop-resize, proprio = pos + axis-angle + qpos.
    img = np.zeros((R.RENDER, R.RENDER, 3), np.uint8)
    img[:32, :32] = 255                                                        # bright top-left block
    th = math.pi / 2                                                           # 90 deg about z
    obs = {"agentview_image": img, "robot0_eye_in_hand_image": img.copy(), "robot0_eef_pos": np.array([1.0, 2, 3]),
           "robot0_eef_quat": np.array([0, 0, math.sin(th / 2), math.cos(th / 2)]),
           "robot0_gripper_qpos": np.array([0.04, -0.04])}
    agent, wrist, state = R.process_obs(obs)
    assert agent.shape == wrist.shape == (224, 224, 3) and agent.dtype == np.uint8
    assert agent[-5:, -5:].min() == 255 and agent[:5, :5].max() == 0          # now bottom-right
    assert np.allclose(state, [1, 2, 3, 0, 0, th, 0.04, -0.04], atol=1e-6) and state.dtype == np.float32
    assert np.allclose(R.quat2axisangle([0, 0, 0, 1.0]), 0)
    print("3 ok: observation processing (rotation, size, proprio)")

    # 4. Gripper: dataset 1 = open -> sim -1 (open); 0 -> +1 (close); binarized; other dims untouched.
    a = np.array([[0.1, -0.2, 0, 0, 0, 0.3, 1.0], [0, 0, 0, 0, 0, 0, 0.0], [0, 0, 0, 0, 0, 0, 0.8],
                  [0, 0, 0, 0, 0, 0, 0.2]], np.float32)
    e = R.to_exec(a)
    assert e[:, -1].tolist() == [-1, 1, -1, 1] and np.array_equal(e[:, :6], a[:, :6])
    print("4 ok: gripper conversion")

    # 5. Noise as FastWAMJoint.infer_action with seed = eval seed: one generator per stream, reseeded each call.
    v, act = R.request_noise(42, 5)
    assert torch.equal(v, torch.randn((1, 5, PW.LATENT_T) + PW.GRID, generator=torch.Generator().manual_seed(42))[0])
    assert torch.equal(act, torch.randn((1, PW.HORIZON, 7), generator=torch.Generator().manual_seed(42))[0])
    assert torch.equal(R.request_noise(42, 5)[1], act) and not torch.equal(R.request_noise(43, 5)[1], act)
    print("5 ok: per-call noise matches Fast-WAM's seeding")

    # 6. Policy on a tiny random model with a fake featurizer: channel assembly, and a batch of requests gives the
    #    same chunks as each request alone (results don't depend on batching).
    torch.manual_seed(0)
    dims = {"vae": 48, "srcA": 6, "srcB": 4}
    in_dim = sum(dims.values())
    kw = {"hidden": 48, "action_hidden": 16, "layers": 1, "heads": 1, "head_dim": 48}
    model = PW.build_model(in_dim, "cpu", **kw)
    cfg = {"config": "srcA+srcB", "seed": 0, "sources": list(dims), "in_dim": in_dim, "model_kw": kw,
           "action_norm": {"lo": [-1.0] * 6 + [0.0], "hi": [1.0] * 6 + [1.0]},
           "state_norm": {"lo": [-2.0] * 8, "hi": [2.0] * 8},
           "feature_stats": {"srcA": [[0.5] * 6, [2.0] * 6], "srcB": [[0.0] * 4, [1.0] * 4]}}

    def featurize(ag, wr):
        B = len(ag)
        m = torch.tensor(ag.reshape(B, -1)[:, :8].astype(np.float32) / 255)
        base = torch.arange(14 * 28, dtype=torch.float32).reshape(14, 28) / 400
        vae = (m[:, :1, None, None, None] + base).expand(B, 48, 1, 14, 28).contiguous()
        return [vae, (m[:, 1:2, None, None] + base[..., None]).expand(B, 14, 28, 6).contiguous(),
                (m[:, 2:3, None, None] - base[..., None]).expand(B, 14, 28, 4).contiguous()]

    with tempfile.TemporaryDirectory() as d:
        g = torch.Generator().manual_seed(1)
        for instr in ["task 0", "task 1"]:
            torch.save({"context": torch.randn(128, 4096, generator=g).to(torch.bfloat16),
                        "mask": torch.arange(128) < 20}, t5_cache.cache_path(d, t5_cache.prompt_of(instr)))
        pol = R.Policy(model, cfg, featurize, d, "cpu")
        rng = np.random.default_rng(0)
        reqs = [{"agent": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                 "wrist": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                 "state": rng.normal(size=8).astype(np.float32), "instruction": f"task {i % 2}", "eval_seed": 41 + i % 3}
                for i in range(5)]
        batch = pol.act(reqs)
        single = np.concatenate([pol.act([r]) for r in reqs])
        assert batch.shape == (5, PW.HORIZON, 7) and np.isfinite(batch).all()
        assert np.abs(batch - single).max() < 1e-4, np.abs(batch - single).max()
        assert set(np.unique(batch[..., -1])) <= {-1.0, 0.0, 1.0}
        assert np.abs(batch[0] - batch[1]).max() > 1e-3                         # different inputs, different chunks
    print(f"6 ok: policy batching is per-item (max diff {np.abs(batch - single).max():.1e})")

    # 7. run_jobs with simulator processes: every job recorded once, identical records with 1 or 3 envs (batching and
    #    worker assignment don't change outcomes), a raising episode goes to the errors file only, reruns resume.
    sub = {"subset_id": "s1", "episodes": [{"suite": "libero_goal", "id": i, "name": f"t{i}", "category": "c"}
                                           for i in [1, 2, 3, 13, 20]]}
    jobs = R.make_jobs(sub, 1, {"libero_goal": ["a"]})
    meta = {"config": "x", "train_seed": 0, "checkpoint": "ck"}
    outs = {}
    with tempfile.TemporaryDirectory() as d:
        for n_envs in (1, 3):
            out = Path(d, f"r{n_envs}.jsonl")
            n_rec, n_err = R.run_jobs(jobs, fake_act, n_envs, out, meta, env_spec="test_rollout:fake_env",
                                      mujoco_gl=None, log=lambda m: None)
            assert (n_rec, n_err) == (len(jobs) - 3, 3), (n_rec, n_err)
            errs = [json.loads(x) for x in Path(str(out) + ".errors.jsonl").read_text().splitlines()]
            assert sorted(e["job"]["eval_seed"] for e in errs) == [41, 42, 43] and "planted failure" in errs[0]["traceback"]
            rows = [json.loads(x) for x in out.read_text().splitlines()]
            outs[n_envs] = sorted((r["benchmark"], r["episode_id"], r["eval_seed"], r["success"], r["steps"]) for r in rows)
            assert all(set(L.FIELDS) <= set(r) for r in rows)
        assert outs[1] == outs[3]
        succ = [o[3] for o in outs[1]]
        assert 0 < sum(succ) < len(succ), succ                                  # both outcomes occur
        # rerun after the errors: nothing new for finished jobs, the raising episode is retried and raises again
        n_rec, n_err = R.run_jobs(jobs, fake_act, 2, Path(d, "r1.jsonl"), meta, env_spec="test_rollout:fake_env",
                                  mujoco_gl=None, log=lambda m: None)
        assert (n_rec, n_err) == (0, 3), (n_rec, n_err)
    print(f"7 ok: run_jobs ({len(jobs)} jobs; same records with 1 or 3 envs; errors kept apart; resumable)")
    print("all checks passed")


if __name__ == "__main__":
    main()
