"""CPU checks for pretrained_train.py on a tiny model (PLAN Hand-off tests 1-5, plus the draws and the sanity modes).
Run from src/: python test_pretrained_train.py

Needs FastWAM 7faa711 importable (pip install --no-deps -e FastWAM) and safetensors. The tiny model and fake Wan
checkpoint are test_pretrained_sizing's (Wan's real tensor names at width 48).

Tolerance for "equal up to floating-point order" (tests 2 and 4), fixed before the first run: relative L2 error
||a - b|| / ||b|| <= 1e-5 over all parameters together (fp32 machine epsilon is 1.2e-7; re-ordering 16 or 32 partial
sums of the same terms is expected to stay far below 1e-5).
"""

import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

import pretrained_sizing as S
import pretrained_train as T
import proxy_wam as P
import rollout as R
from test_pretrained_sizing import SHARD1, SHARD2, TINY, fake_wan
from test_proxy_wam import fake_cache

TOL = 1e-5
SUITES = "suite_a,suite_b"
LENGTHS = (200, 150, 250)       # 600 windows per suite, 1,200 in all: 9 batches of 128 per epoch


def rel(a, b):
    a = torch.cat([x.reshape(-1).double() for x in a])
    b = torch.cat([x.reshape(-1).double() for x in b])
    return float((a - b).norm() / b.norm())


def argv(d, out, *extra, micro=8, gb=32):
    return ["--feats", str(d), "--t5", str(d / "t5"), "--suites", SUITES, "--wan-dir", str(d / "wan"), "--keep", "0-1",
            "--hidden", "48", "--action-hidden", "16", "--heads", "2", "--head-dim", "24", "--device", "cpu",
            "--global-batch", str(gb), "--micro-batch", str(micro), "--allow-any-micro-batch", "--buffer-episodes",
            "16", "--read-threads", "2", "--assemble-threads", "1", "--epochs", "1", "--out", str(d / out), *extra]


def run(a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = T.main(a)
    return rc, buf.getvalue()


def model_pt(d, out):
    return torch.load(d / out / "model.pt", map_location="cpu", weights_only=False)


def main():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fake_cache(d, SUITES.split(","), [], np.random.default_rng(0), n_eps=3, lengths=LENGTHS)
        fake_wan(d / "wan", n_blocks=4)
        data = P.Data(d, SUITES.split(","), [], d / "t5")
        data.context_dev = data.context
        lengths = [n for _, _, n, _ in data.episodes]
        model = S.build(data.in_dim, "cpu", 2, TINY, seed=0)

        # 1. Draws. t_from_u on a generator's uniforms equals Fast-WAM's sample_training_t on the global RNG with the
        #    same seed (both schedulers); draw_step depends only on (seed, step); training_loss with draws=None equals
        #    training_loss given the same draws made in the same order from the global RNG (the proxy_wam edit keeps
        #    the old behaviour).
        for sch in (model.train_video_scheduler, model.train_action_scheduler):
            torch.manual_seed(5)
            ref = sch.sample_training_t(batch_size=64, device="cpu", dtype=torch.float32)
            got = T.t_from_u(sch, torch.rand((64,), generator=torch.Generator().manual_seed(5), dtype=torch.float32))
            assert torch.equal(ref, got)
        ls = T.lat_shape_of(data)
        a1, a2 = T.draw_step(model, 0, 7, 128, ls), T.draw_step(model, 0, 7, 128, ls)
        assert all(torch.equal(x, y) for x, y in zip(a1, a2))
        assert not torch.equal(a1[0], T.draw_step(model, 0, 8, 128, ls)[0])
        assert not torch.equal(a1[0], T.draw_step(model, 1, 7, 128, ls)[0])
        assert a1[0].shape == (128,) + ls and a1[2].shape == (128, P.HORIZON, 7) and a1[1].shape == a1[3].shape == (128,)
        assert float(a1[1].min()) >= 0 and float(a1[1].max()) <= 1000 and float(a1[3].max()) <= 1000
        plan = P.make_plan(lengths, 1, 128, 16, 0)
        hb = data.assemble(plan[0], {k: {P.VAE: data.load_file(k, P.VAE)} for k in
                                     {data.episodes[ei][:2] for ei in np.unique(plan[0][:, 0])}})
        bt = P.to_device(T.host_slice(hb, slice(0, 8)), data, "cpu")
        sl = data.channel_slices()
        with torch.no_grad():
            torch.manual_seed(3)
            l_old, _ = P.training_loss(model, bt, sl)
            torch.manual_seed(3)
            x, act = bt["latents"], bt["action"]
            dr = (torch.randn_like(x), model.train_video_scheduler.sample_training_t(8, "cpu", x.dtype),
                  torch.randn_like(act), model.train_action_scheduler.sample_training_t(8, "cpu", act.dtype))
            l_new, _ = P.training_loss(model, bt, sl, draws=dr)
        assert torch.equal(l_old, l_new), (l_old, l_new)
        print("1 ok: draws keyed by (seed, step), same distribution code as Fast-WAM's sample_training_t; "
              "training_loss unchanged without draws")

        # 2. One optimizer step's gradient: micro-batches 8 x 16 and 4 x 32 equal one call on all 128 with the same
        #    draws, within TOL; the loss too.
        draws = T.draw_step(model, 0, 0, 128, ls)
        grads, losses = {}, {}
        for m in (128, 8, 4):
            model.zero_grad(set_to_none=True)
            losses[m], _ = T.forward_backward(model, hb, data, draws, m, "cpu")
            grads[m] = [p.grad.clone() for p in model.parameters()]
        e8, e4 = rel(grads[8], grads[128]), rel(grads[4], grads[128])
        assert e8 <= TOL and e4 <= TOL, (e8, e4)
        assert abs(losses[8] - losses[128]) <= TOL * abs(losses[128]) and abs(losses[4] - losses[128]) <= TOL * abs(losses[128])
        print(f"2 ok: accumulated gradient equals the batch of 128 (rel. L2 error 8x16 {e8:.1e}, 4x32 {e4:.1e}; "
              f"tolerance {TOL:.0e})")

        # 5 (PLAN test 4). Kept blocks equal the Wan file after build + load + action-expert derivation; the
        #    derivation leaves the video expert as loaded.
        args = T.parser().parse_args(argv(d, "unused"))
        m0, rep = T.build_and_init(data, args, "cpu", lambda m: None)
        wan = {**load_file(str(d / "wan" / SHARD1)), **load_file(str(d / "wan" / SHARD2))}
        vs = m0.video_expert.state_dict()
        kept = [k for k in wan if not k.startswith("blocks.") or int(k.split(".")[1]) < 2]
        assert set(kept) == set(vs), set(kept) ^ set(vs)
        assert all(torch.equal(vs[k], wan[k]) for k in kept), [k for k in kept if not torch.equal(vs[k], wan[k])]
        c = rep["counts_by_tag"]
        assert c == {"loaded": len(kept), "partly loaded": 0, "derived": 64, "fresh": 4 + 2}, c
        assert rep["counts"]["derived"]["copied"] == 20 and rep["counts"]["derived"]["resized"] == 44, rep["counts"]
        assert sum(1 for e in rep["parameters"].values() if e["tag"] == "derived") == 64
        assert all(e["tag"] == "new" for n, e in rep["parameters"].items()
                   if n.startswith(("action_expert.action_encoder.", "action_expert.head.", "proprio_encoder.")))
        assert "counts_before_derivation" in rep
        w0 = [p.detach().clone() for p in m0.parameters()]
        print(f"5 ok: kept blocks equal the Wan file ({len(kept)} tensors); load report {c}")

        # 3 (PLAN test 2). Straight run of 4 steps vs 2 steps, lost pod, resume at the same micro-batch: bit for bit
        #    (weights, losses). ckpt.pt is kept at the end.
        rc, _ = run(argv(d, "A", "--max-steps", "4", "--ckpt-every", "2", "--first-batch-report"))
        assert rc == 0
        rc, logB0 = run(argv(d, "B", "--max-steps", "4", "--ckpt-every", "2", "--stop-after", "2",
                             "--first-batch-report"))
        assert "[projection] with checkpoint writes: first write took" in logB0 and "+ 18 writes x" in logB0 and "+ 0 writes =" in logB0, logB0
        assert rc == 3 and (d / "B" / "ckpt.pt").exists() and not (d / "B" / "model.pt").exists()
        assert torch.load(d / "B" / "ckpt.pt", weights_only=False)["step"] == 2
        fb_before = (d / "B" / "first-batch.json").read_bytes()
        rc, logB = run(argv(d, "B", "--max-steps", "4", "--ckpt-every", "2", "--first-batch-report"))
        assert rc == 0 and "resuming" in logB and "MICRO-BATCH CHANGED" not in logB
        assert "[first batch] skipped" in logB and (d / "B" / "first-batch.json").read_bytes() == fb_before
        A, B = model_pt(d, "A"), model_pt(d, "B")
        assert all(torch.equal(A["model"][k], B["model"][k]) for k in A["model"])
        hA = json.loads((d / "A" / "train.json").read_text())["history"]
        hB = json.loads((d / "B" / "train.json").read_text())["history"]
        assert [h["step"] for h in hB] == [0, 1, 2, 3] and [h["loss"] for h in hA] == [h["loss"] for h in hB]
        assert (d / "A" / "ckpt.pt").exists() and (d / "B" / "ckpt.pt").exists()
        try:
            run(argv(d, "B", "--max-steps", "4", "--ckpt-every", "2", "--epochs", "2"))
            raise AssertionError("a resume with a different plan was accepted")
        except SystemExit as e:
            assert "different run" in str(e) and "epochs" in str(e), e
        try:
            run(argv(d, "B", "--max-steps", "2", "--ckpt-every", "2"))
            raise AssertionError("a resume at or past --max-steps was accepted")
        except SystemExit as e:
            assert "at or past the end" in str(e), e
        print("3 ok: resume at the same micro-batch reproduces the straight run bit for bit; ckpt.pt kept; first-batch "
              "report kept on resume; projection re-logged with the write time; a resume with a different plan or "
              "past --max-steps is refused")

        # 4 (PLAN test 3). Lost pod at step 2 with micro-batch 8, resumed at micro-batch 4: matches the straight run
        #    within TOL (on the weights' change from their initial values); the change is logged and recorded.
        rc, _ = run(argv(d, "C", "--max-steps", "4", "--ckpt-every", "2", "--stop-after", "2", micro=8))
        assert rc == 3
        rc, logC = run(argv(d, "C", "--max-steps", "4", "--ckpt-every", "2", micro=4))
        assert rc == 0 and "MICRO-BATCH CHANGED at step 2: 8 -> 4" in logC
        Cm = model_pt(d, "C")
        names = [n for n, _ in m0.named_parameters()]
        dA = [A["model"][n] - w for n, w in zip(names, w0)]
        dC = [Cm["model"][n] - w for n, w in zip(names, w0)]
        e = rel(dC, dA)
        assert e <= TOL, e
        assert not all(torch.equal(A["model"][n], Cm["model"][n]) for n in names)    # it did run another path
        tc = json.loads((d / "C" / "train.json").read_text())
        assert tc["micro_batch_history"] == [{"step": 0, "micro_batch": 8}, {"step": 2, "micro_batch": 4}], tc
        print(f"4 ok: resume at another micro-batch matches the straight run (rel. L2 error of the weight change "
              f"{e:.1e}); change logged and recorded")

        # 6 (PLAN test 5). rollout.Policy.load builds the new model from the saved model.pt; parameters match.
        orig = R.Featurizer
        R.Featurizer = lambda *a, **k: None          # the VAE is not needed to build the policy
        try:
            pol = R.Policy.load(d / "A", d / "t5", None, "cpu")
        finally:
            R.Featurizer = orig
        sd = pol.model.state_dict()
        assert set(sd) == set(A["model"]) and all(torch.equal(sd[k], A["model"][k]) for k in sd)
        assert A["config"]["model_kw"] == {**TINY, "layers": 2}, A["config"]["model_kw"]
        assert len(pol.model.video_expert.blocks) == len(pol.model.action_expert.blocks) == 2
        print(f"6 ok: rollout.Policy.load builds the trained model from model.pt (model_kw {A['config']['model_kw']})")

        # 7. Sanity modes: first-batch report (sanity checks 2-3), overfit one batch (check 1), stop on a
        #    non-finite loss.
        fb = json.loads((d / "A" / "first-batch.json").read_text())
        assert fb["batch_size"] == 32 and len(fb["tasks"]) == 32 and "predict_zero" in fb["loss_at_start"]
        assert set(fb["loss_at_start"]["model"]) == {"total", "vae", "action"}
        assert fb["inputs"]["latents_host"]["shape"] == [32, P.LATENT_T, *P.GRID, 48]
        rc, _ = run(argv(d, "O", "--overfit-steps", "60", "--overfit-lr", "1e-3"))
        ov = json.loads((d / "O" / "overfit.json").read_text())
        first, last = ov["first_6_mean"], ov["last_6_mean"]
        assert rc == 0 and len(ov["history"]) == 60 and last["loss/action"] < first["loss/action"], (first, last)
        assert not (d / "O" / "model.pt").exists() and not (d / "O" / "ckpt.pt").exists()
        orig = P.training_loss
        P.training_loss = lambda *a, **k: (lambda l, p: (l * float("nan"), p))(*orig(*a, **k))
        try:
            rc, logN = run(argv(d, "N", "--max-steps", "2", "--ckpt-every", "1"))
        finally:
            P.training_loss = orig
        assert rc == 2 and "STOP" in logN and not (d / "N" / "model.pt").exists() and not (d / "N" / "ckpt.pt").exists()
        assert "NonFiniteLoss" in json.loads((d / "N" / "train.json").read_text())["stopped"]
        P.training_loss = lambda *a, **k: (lambda l, p: (l * float("nan"), p))(*orig(*a, **k))
        try:
            rc1, _ = run(argv(d, "N1", "--max-steps", "2", "--first-batch-report"))
            rc2, _ = run(argv(d, "N2", "--overfit-steps", "3"))
        finally:
            P.training_loss = orig
        assert rc1 == 2 and "NonFiniteLoss" in json.loads((d / "N1" / "first-batch.json").read_text())["stopped"]
        assert rc2 == 2 and "NonFiniteLoss" in json.loads((d / "N2" / "overfit.json").read_text())["stopped"]
        print(f"7 ok: first-batch report written; overfit mode (action loss {first['loss/action']:.3f} -> "
              f"{last['loss/action']:.3f}); a non-finite loss stops (training, first-batch report, overfit) before any "
              f"step or checkpoint")

        # 8. Step watchdog: not armed before the first step; fires once with exit code 17 and a log line when no step
        #    finishes in time (a fake slow step); a normal run is unaffected and changes nothing in the saved weights.
        import time
        codes, logs = [], []
        wd = T.StepWatchdog(0.4, logs.append, exit_fn=codes.append)
        time.sleep(0.8)
        assert codes == [], "fired before the first step finished"
        wd.beat()
        time.sleep(0.2)
        assert codes == [], "fired early"
        time.sleep(1.0)
        wd.stop()
        assert codes == [T.WATCHDOG_EXIT] and "[watchdog]" in logs[0], (codes, logs)
        # first deadline: armed from construction; fires in the start-up phase if no step finishes; after the first
        # beat the (longer) step limit applies instead
        codes.clear(); logs.clear()
        wd = T.StepWatchdog(30, logs.append, exit_fn=codes.append, first_s=0.4)
        time.sleep(1.0)
        wd.stop()
        assert codes == [T.WATCHDOG_EXIT] and "start-up" in logs[0], (codes, logs)
        codes.clear(); logs.clear()
        wd = T.StepWatchdog(30, logs.append, exit_fn=codes.append, first_s=0.4)
        time.sleep(0.2)
        wd.beat()
        time.sleep(0.8)
        wd.stop()
        assert codes == [] and logs == [], (codes, logs)
        # R1: logging blocks forever; the process still exits (timer started before the log call)
        codes.clear()
        import threading
        block = threading.Event()
        wd = T.StepWatchdog(0.3, lambda m: block.wait(), exit_fn=codes.append, kill_after_s=0.3)
        wd.beat()
        time.sleep(1.5)
        wd.stop()
        assert codes == [T.WATCHDOG_EXIT], codes
        block.set()
        # R4: history.jsonl with a cut-off last line is read, the bad line skipped and counted
        hp = d / "hist-test.jsonl"
        hp.write_text('{"step": 0, "loss": 1.0}\n{"step": 1, "loss": 2.0}\n{"step": 2, "lo')
        hl = []
        got = T.read_history(hp, hl.append)
        assert [r["step"] for r in got] == [0, 1] and "skipped 1 unparseable" in hl[0], (got, hl)
        wd = T.StepWatchdog(0, logs.append, exit_fn=codes.append)
        assert wd.thread is None
        codes.clear()
        slow, orig_step = [], T.optimizer_step
        def slow_step(*a, **k):
            if not slow:
                slow.append(1)
            elif len(slow) == 1:
                slow.append(2)
                time.sleep(1.5)
            return orig_step(*a, **k)
        orig_exit, T._watchdog_exit = T._watchdog_exit, codes.append
        T.optimizer_step = slow_step
        try:
            rc, logW = run(argv(d, "W", "--max-steps", "3", "--ckpt-every", "2", "--step-watchdog-s", "0.6"))
        finally:
            T.optimizer_step = orig_step
            T._watchdog_exit = orig_exit
        assert rc == 0 and codes == [T.WATCHDOG_EXIT] and "[watchdog] fired in the training steps phase: no progress for 1 s" in logW, (codes, logW)
        codes.clear()
        rc, logA2 = run(argv(d, "A2", "--max-steps", "4", "--ckpt-every", "2"))
        assert rc == 0 and codes == [] and "[watchdog]" not in logA2
        A2 = model_pt(d, "A2")
        assert all(torch.equal(A["model"][k], A2["model"][k]) for k in A["model"])
        print("8 ok: the step watchdog is not armed before the first step, fires once with exit 17 and a log line on a "
              "slow step, and a normal run is unaffected (same weights as the straight run)")
    print("all checks passed")


if __name__ == "__main__":
    main()
