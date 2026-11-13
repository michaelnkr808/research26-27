"""Synthetic CPU checks for pretrained_sizing.py. Run from src/: python test_pretrained_sizing.py (or pytest on it).

Needs FastWAM 7faa711 importable (pip install --no-deps -e FastWAM) and safetensors. A tiny model and a fake Wan
checkpoint written with Wan's real tensor names (WanVideoDiT's own state_dict names, which equal the checkpoint's,
step2-reading.md Q2; check 1 confirms the full-size name list against the index facts recorded there).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

import pretrained_sizing as S
import proxy_wam as P
from test_proxy_wam import fake_cache

TINY = {"hidden": 48, "action_hidden": 16, "heads": 2, "head_dim": 24}
EXTRA = 8
SHARD1, SHARD2 = "diffusion_pytorch_model-00001-of-00002.safetensors", "diffusion_pytorch_model-00002-of-00002.safetensors"


def fake_wan(root, n_blocks=4, split=2, edit=None):
    """A Wan-style checkpoint folder: config.json, index json, two shards (shard 1: blocks < split and every non-block
    tensor, as Wan's shard 1; shard 2: the other blocks). Values random, not the sizing model's own init.
    edit(sd) may change the tensors before writing. Returns the tensors written."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(123)
    with torch.device("cpu"):
        src = P.build_model(48, "cpu", layers=n_blocks, **TINY).video_expert
    sd = {k: v.detach().clone() + 0.01 * torch.randn_like(v) for k, v in src.state_dict().items()}
    if edit:
        edit(sd)
    in1 = {k: v for k, v in sd.items() if not k.startswith("blocks.") or int(k.split(".")[1]) < split}
    in2 = {k: v for k, v in sd.items() if k not in in1}
    save_file({k: v.contiguous() for k, v in in1.items()}, str(root / SHARD1))
    save_file({k: v.contiguous() for k, v in in2.items()}, str(root / SHARD2))
    wm = {**{k: SHARD1 for k in in1}, **{k: SHARD2 for k in in2}}
    (root / S.WAN_INDEX).write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": wm}))
    (root / S.WAN_CONFIG).write_text(json.dumps({"dim": 48, "ffn_dim": round(48 * 14336 / 3072), "freq_dim": 256,
                                                 "in_dim": 48, "num_heads": 2, "num_layers": n_blocks, "out_dim": 48}))
    return sd


def expect_load_error(model, d, keep, what):
    try:
        S.load_wan(model, d, keep, log=lambda m: None)
    except S.LoadError as e:
        return str(e)
    raise AssertionError(f"{what} not caught")


def check_full_size_names():
    """1. The full-size 30-layer video expert built as the sizing script builds it has exactly the checkpoint's 825
    tensor names (step2-reading.md Q1) with Wan's naming, and the patch embedding has the checkpoint's shape."""
    # build_model's own WanVideoDiT / ActionDiT / MoT calls on the meta device (no memory); FastWAMJoint's scheduler
    # setup can't run on meta, so it is replaced by a holder for the two experts for this check only
    from types import SimpleNamespace
    joint = P.FastWAMJoint
    P.FastWAMJoint = lambda video, action, mot, **kw: SimpleNamespace(video_expert=video, action_expert=action)
    try:
        with torch.device("meta"):
            m = P.build_model(48, "meta", layers=S.WAN_LAYERS, **S.WAN_KW)
    finally:
        P.FastWAMJoint = joint
    names = dict(m.video_expert.named_parameters())
    assert len(names) == 825, len(names)
    for n in ["blocks.0.self_attn.q.weight", "blocks.29.ffn.0.weight", "blocks.5.norm3.weight", "blocks.5.modulation",
              "head.head.weight", "head.modulation", "patch_embedding.weight", "patch_embedding.bias",
              "text_embedding.0.weight", "text_embedding.2.bias", "time_embedding.0.weight", "time_projection.1.weight"]:
        assert n in names, n
    assert tuple(names["patch_embedding.weight"].shape) == (3072, 48, 1, 2, 2)
    assert tuple(names["head.head.weight"].shape) == (48 * 4, 3072)
    assert m.action_expert.hidden_dim == 1024 and m.action_expert.ffn_dim == 4096 and names["blocks.0.ffn.0.weight"].shape[0] == 14336
    print("1 ok: full-size 30-layer video expert has the checkpoint's 825 names and shapes")


def check_load(d):
    """2. Keep blocks 0-1 with 8 extra channels: kept blocks equal the source tensors (diff 0), widened projections'
    48-channel slices equal the source and the extra slices are zero, dropped blocks skipped, only shard 1 read."""
    src = fake_wan(d)
    (Path(d) / SHARD2).rename(Path(d) / (SHARD2 + ".hidden"))      # shard 2 must not be needed for blocks 0-1
    model = S.build(48 + EXTRA, "cpu", 2, TINY, seed=0)
    rep = S.load_wan(model, d, [0, 1], log=lambda m: None)
    (Path(d) / (SHARD2 + ".hidden")).rename(Path(d) / SHARD2)
    file_sd = load_file(str(Path(d) / SHARD1))
    params = dict(model.named_parameters())
    tags = {n: e["tag"] for n, e in rep["parameters"].items()}
    assert rep["shards_read"] == [SHARD1] and rep["shards_not_read"] == [SHARD2]
    assert set(tags) == set(params), "every model parameter is tagged"
    for n, e in rep["parameters"].items():
        if e["tag"] == "loaded":
            assert e["max_abs_diff"] == 0 and torch.equal(params[n].detach(), file_sd[e["source"]]), n
            assert torch.equal(params[n].detach(), src[e["source"]]), n
    assert {n for n, t in tags.items() if t == "partly loaded"} == {"video_expert." + w for w in S.WIDENED}
    assert all(t == "new" for n, t in tags.items() if not n.startswith("video_expert."))
    assert all(n.startswith(("video_expert.", "action_expert.", "proprio_encoder.")) for n in tags)
    assert all(t != "new" for n, t in tags.items() if n.startswith("video_expert."))
    pw = params["video_expert.patch_embedding.weight"].detach()
    assert pw.shape[1] == 48 + EXTRA and torch.equal(pw[:, :48], src["patch_embedding.weight"]) and (pw[:, 48:] == 0).all()
    for k in ("weight", "bias"):
        hw = params[f"video_expert.head.head.{k}"].detach()
        hv = hw.view((4, 48 + EXTRA) + tuple(hw.shape[1:]))
        assert torch.equal(hv[:, :48], src[f"head.head.{k}"].view((4, 48) + tuple(hw.shape[1:]))) and (hv[:, 48:] == 0).all()
    sk = rep["skipped"]
    n_per_block = sum(1 for k in src if k.startswith("blocks.0."))
    assert len(sk) == 2 * n_per_block and {e["source"].split(".")[1] for e in sk} == {"2", "3"}
    assert all(e["tag"] == "skipped" and "not kept" in e["reason"] for e in sk)
    c = rep["counts"]
    assert c["loaded"]["elements"] + c["partly loaded"]["elements"] + c["new"]["elements"] == rep["total_params"]
    assert c["partly loaded"]["new_elements"] == EXTRA * 48 * 4 + EXTRA * 4 * 48 + EXTRA * 4
    assert all(e.get("checked_against", "").startswith("tensor re-read") for e in rep["parameters"].values()
               if e["tag"] != "new")
    # the check compares with the file, not with what was copied: a changed loaded value or a nonzero extra slice
    # after loading is caught
    ents = rep["parameters"]
    with torch.no_grad():
        p = params["video_expert.blocks.1.ffn.0.weight"]
        p[0, 0] += 1e-6
        try:
            S.verify_against_file(model, d, ents)
            raise AssertionError("changed loaded value not caught")
        except S.LoadError:
            pass
        p[0, 0] = file_sd["blocks.1.ffn.0.weight"][0, 0]
        S.verify_against_file(model, d, ents)
        params["video_expert.patch_embedding.weight"][0, 48] = 0.5
        try:
            S.verify_against_file(model, d, ents)
            raise AssertionError("nonzero extra slice not caught")
        except S.LoadError:
            pass
        params["video_expert.patch_embedding.weight"][0, 48] = 0.0
    print(f"2 ok: load report (loaded {c['loaded']['tensors']}, partly {c['partly loaded']['tensors']}, "
          f"new {c['new']['tensors']}, skipped {len(sk)}), only shard 1 read, checked against the file")
    return model


def check_widened_equivalence(d):
    """3. With the zero extra slices, the widened model predicts exactly what the 48-channel model does on the VAE
    channels (whatever the extra input channels hold) and 0 on the extra channels: it starts as the pretrained one."""
    narrow = S.build(48, "cpu", 2, TINY, seed=0)
    wide = S.build(48 + EXTRA, "cpu", 2, TINY, seed=0)
    rn = S.load_wan(narrow, d, [0, 1], log=lambda m: None)
    S.load_wan(wide, d, [0, 1], log=lambda m: None)
    assert all(e["tag"] in ("loaded", "new") for e in rn["parameters"].values()), "48-channel model: nothing partial"
    with torch.no_grad():
        for n, p in wide.named_parameters():
            if not n.startswith("video_expert."):
                p.copy_(dict(narrow.named_parameters())[n])
    B = 2
    torch.manual_seed(7)
    lat = torch.randn(B, 48 + EXTRA, P.LATENT_T, *P.GRID)
    kw = dict(latents_action=torch.randn(B, P.HORIZON, 7), timestep_video=torch.rand(B) * 1000,
              timestep_action=torch.rand(B) * 1000, context=torch.randn(B, 129, 4096),
              context_mask=torch.ones(B, 129, dtype=torch.bool), fuse_vae_embedding_in_latents=True)
    with torch.no_grad():
        vn, an = narrow._joint_denoise_core(latents_video=lat[:, :48], attention_mask=P._mask(narrow, B, "cpu"), **kw)
        vw, aw = wide._joint_denoise_core(latents_video=lat, attention_mask=P._mask(wide, B, "cpu"), **kw)
    assert vw.shape[1] == 48 + EXTRA and (vw[:, 48:] == 0).all()
    assert torch.allclose(vw[:, :48], vn, atol=1e-5) and torch.allclose(aw, an, atol=1e-5), \
        (float((vw[:, :48] - vn).abs().max()), float((aw - an).abs().max()))
    print("3 ok: widened model equals the 48-channel model on VAE channels, predicts 0 on extra channels")


def check_other_keep(d):
    """4. A non-leading keep set maps checkpoint blocks to model positions (blocks.1 -> 0, blocks.3 -> 1) and reads
    both shards."""
    src = fake_wan(d)
    model = S.build(48, "cpu", 2, TINY, seed=0)
    rep = S.load_wan(model, d, [1, 3], log=lambda m: None)
    sd = model.video_expert.state_dict()
    for k in [k for k in src if k.startswith("blocks.1.")]:
        assert torch.equal(sd[k.replace("blocks.1.", "blocks.0.", 1)], src[k]), k
    for k in [k for k in src if k.startswith("blocks.3.")]:
        assert torch.equal(sd[k.replace("blocks.3.", "blocks.1.", 1)], src[k]), k
    assert rep["shards_read"] == [SHARD1, SHARD2] and {e["source"].split(".")[1] for e in rep["skipped"]} == {"0", "2"}
    print("4 ok: keep [1, 3] maps blocks by position and reads the shards it needs")


def check_load_errors(root):
    """5. An unexpected key, a shape mismatch in a kept block, a missing tensor, a config mismatch, a missing needed
    shard, or a keep set past the checkpoint's blocks stops the load."""
    def add(sd):
        sd["foo.weight"] = torch.zeros(3)
    model = S.build(48, "cpu", 2, TINY, seed=0)
    d = Path(root) / "unexpected"
    fake_wan(d, edit=add)
    assert "no model parameter" in expect_load_error(model, d, [0, 1], "unexpected key")

    def bad_shape(sd):
        sd["blocks.1.ffn.0.weight"] = torch.zeros(5, 48)
    d = Path(root) / "shape"
    fake_wan(d, edit=bad_shape)
    assert "shape mismatch" in expect_load_error(model, d, [0, 1], "shape mismatch")

    def drop(sd):
        del sd["blocks.1.norm3.weight"]
    d = Path(root) / "missing"
    fake_wan(d, edit=drop)
    assert "no checkpoint tensor" in expect_load_error(model, d, [0, 1], "missing tensor")

    d = Path(root) / "config"
    fake_wan(d)
    cfg = json.loads((d / S.WAN_CONFIG).read_text())
    (d / S.WAN_CONFIG).write_text(json.dumps({**cfg, "dim": 96}))
    assert "config.json" in expect_load_error(model, d, [0, 1], "config mismatch")

    d = Path(root) / "noshard"
    fake_wan(d)
    (d / SHARD2).unlink()
    assert "not in" in expect_load_error(model, d, [0, 3], "missing shard")
    assert "past the checkpoint" in expect_load_error(model, d, [0, 4], "keep past the end")
    print("5 ok: unexpected key, shape mismatch, missing tensor, config mismatch, missing shard all stop the load")


def check_classify():
    """6. ok / oom / error: only torch.cuda.OutOfMemoryError is oom; a RuntimeError saying 'out of memory' is not."""
    assert S.classify(None) == {"outcome": "ok"}
    r = S.classify(torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert r["outcome"] == "oom" and r["error_type"] == "OutOfMemoryError" and "Tried to allocate" in r["error_message"]
    r = S.classify(RuntimeError("shape '[2, 3]' is invalid"))
    assert r["outcome"] == "error" and r["error_type"] == "RuntimeError" and "traceback" in r
    assert S.classify(RuntimeError("CUDA error: out of memory"))["outcome"] == "error"
    assert S.classify(S.LoadError("x"))["outcome"] == "error"
    print("6 ok: ok / oom / error classification")


def check_m_star():
    """7. m* = largest power of two with an ok sweep row and peak reserved <= 0.9 M. Fit list: the peak at m is the
    largest of the sweep peak, the sweep's own full-step peak and a global-batch run's peak at m; an oom in the
    global-batch run at m* means m* does not fit and the m*/2 retry decides; t_step only at an m that fits, else
    flagged; the linear fit uses sweep peaks only and the update's extra memory is a separate constant."""
    M = 1000
    row = lambda m, pk, out="ok": {"m": m, "peak_reserved_bytes": pk, "outcome": out}
    rows = [row(1, 500), row(2, 700), row(4, 900), row(8, 901)]
    assert S.m_star(rows, M) == 4                                    # 900 <= 0.9 x 1000; 901 is not
    assert S.m_star(rows + [row(16, None, "oom")], M) == 4
    assert S.m_star([row(1, 950)], M) is None
    assert S.m_star([row(1, 100), row(3, 200)], M) == 1               # 3 is not a power of two
    assert S.m_star([row(1, 100), row(2, 100, "error")], M) == 1
    assert S.m_star([row(1, 100), row(2, 100)], M, global_batch=1) == 1
    assert S.m_star([row(1, 100), row(2, 999)], None) == 2            # CPU smoke: largest ok m

    def srow(m, pk, full):
        return {**row(m, pk), "full_step_peak_reserved_bytes": full}

    def orow(m, pk, out="ok", sec=5.0):
        return {**row(m, pk, out), "sec_per_opt_step": {"mean": sec}, "global_batch": 128}

    sweep = [srow(1, 500, 560), srow(2, 700, 760), srow(4, 850, 880), srow(8, 901, None)]
    slope = np.polyfit([1, 2, 4, 8], [500, 700, 850, 901], 1)[0]
    # A: global-batch run at m* = 4 ok but above the limit: 4 does not fit, no t_step at an m that fits (flagged)
    fit = S.fit_summary(sweep, [orow(4, 950)], M, 4)
    assert fit["peak_reserved_by_m"][4]["peak_reserved_bytes"] == 950 and fit["largest_m_that_fits"] == 2
    assert fit["t_step"] is None and "no full-optimizer-step time" in fit["t_step_flag"]
    assert abs(fit["linear_fit_sweep_peak_vs_m"]["bytes_per_sample"] - slope) < 1e-6
    ex = fit["optimizer_update_extra_bytes"]
    assert ex["by_m_sweep_full_step"] == {1: 60, 2: 60, 4: 30} and ex["by_m_global_batch_steps"] == {4: 100}
    assert ex["constant"] == 100
    # B: oom at m* = 4, the m*/2 retry ok within the limit: 4 does not fit (oom), t_step from m = 2
    fit = S.fit_summary(sweep, [orow(4, None, "oom"), orow(2, 800, sec=3.0)], M, 4)
    assert fit["largest_m_that_fits"] == 2 and "out of memory" in fit["does_not_fit"][4]
    assert fit["t_step"] == {"m": 2, "sec_per_opt_step": 3.0, "global_batch": 128}
    assert fit["peak_reserved_by_m"][2]["peak_reserved_bytes"] == 800
    # C: global-batch run at m* = 4 ok within the limit: 4 fits, t_step from 4
    fit = S.fit_summary(sweep, [orow(4, 890)], M, 4)
    assert fit["largest_m_that_fits"] == 4 and fit["t_step"]["m"] == 4 and fit["fits_at_m1"]
    # D: the sweep's own full-step peak alone can push m over the limit
    fit = S.fit_summary([srow(1, 500, 560), srow(2, 880, 905)], [], M, 2)
    assert fit["largest_m_that_fits"] == 1 and fit["does_not_fit"][2] == "peak above 0.9 x total memory"
    print("7 ok: m* rule and fit list (full-step peaks, oom at m*, t_step only where it fits, update extra)")


def check_sanity(feats):
    """8. Sanity on a tiny model: finite loss with every part, predict-zero baseline on the same draws (the model's own
    method restored after), inputs summarised; a NaN prediction stops the check."""
    data, _ = S.load_data(feats, "dinov3")
    data.context_dev = data.context
    cache, _ = S.read_all(data)
    hb = S.host_batches(data, cache, 4, 1, 0)[0][0]
    model = S.build(data.in_dim, "cpu", 2, TINY, seed=0)
    bt = P.to_device(hb, data, "cpu")
    rep = S.sanity(model, hb, bt, data, 0, amp=False)
    lm, lz = rep["loss_at_start"]["model"], rep["loss_at_start"]["predict_zero"]
    assert set(lm) == {"total", "vae", "dinov3", "action"} == set(lz)
    assert lm != lz and "_joint_denoise_core" not in model.__dict__
    # same draws: the model's loss recomputed with seed + 1 equals the reported one
    torch.manual_seed(1)
    with torch.no_grad():
        again, _ = P.training_loss(model, bt, data.channel_slices())
    assert abs(float(again) - lm["total"]) < 1e-6, (float(again), lm["total"])
    assert abs(rep["channels/dinov3"]["mean_range"][0]) < 3 and rep["inputs"]["latents"]["shape"][1] == data.in_dim
    assert len(rep["tasks"]) == 4 and rep["padded_action_steps"] >= 0
    model._joint_denoise_core = lambda **kw: (torch.full_like(kw["latents_video"], float("nan")),
                                              torch.zeros_like(kw["latents_action"]))
    try:
        # the model's NaN output is evaluated before sanity() swaps in the predict-zero stand-in
        S.sanity(model, hb, bt, data, 0, amp=False)
        raise AssertionError("NaN loss not caught")
    except S.SanityError:
        pass
    print(f"8 ok: sanity (loss {lm['total']:.3f} vs predict-zero {lz['total']:.3f}), NaN stops the check")


def run_launch(feats, wan, out, state, configs="none,dinov3"):
    cmd = [sys.executable, "pretrained_sizing.py", "launch", "--feats", str(feats), "--wan-dir", str(wan),
           "--out", str(out), "--state-dir", str(state), "--device", "cpu", "--configs", configs, "--ms", "1,2",
           "--global-batch", "4", "--sanity-batch", "4", "--keep", "0-1", "--hidden", "48", "--action-hidden", "16",
           "--heads", "2", "--head-dim", "24"]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)


def check_launcher(root, feats):
    """9. End to end on CPU: launcher runs one process per m and the optimizer-step process at m*, writes load, sanity
    and measure JSONs, removes the saved state; a broken checkpoint ends as error at the first process and stops the
    sweep; an existing results file is never overwritten."""
    wan = Path(root) / "wan"
    fake_wan(wan)
    out, state = Path(root) / "results", Path(root) / "state"
    r = run_launch(feats, wan, out, state)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for cfg, in_dim in (("none", 48), ("dinov3", 56)):
        res = json.loads((out / f"measure-{cfg}.json").read_text())
        assert [x["m"] for x in res["rows"]] == [1, 2] and all(x["outcome"] == "ok" for x in res["rows"]), res["rows"]
        assert res["rows"][0]["from_wan"] and not res["rows"][1]["from_wan"]
        assert res["m_star"] == 2 and res["optimizer_steps"][0]["outcome"] == "ok"
        assert res["optimizer_steps"][0]["accumulation"] == 2 and len(res["optimizer_steps"][0]["sec_per_opt_step"]["timed"]) == 2
        assert res["keep"] == [0, 1] and res["keep_tag"] == S.KEEP_TAG and res["fastwam_commit"] == S.FASTWAM_COMMIT
        assert res["rows"][1]["in_dim"] == in_dim and len(res["rows"][1]["sec_per_micro_step"]["all"]) == S.TIMED
        assert "host_assembly_s_per_batch" in res["rows"][0] and "phase_seconds" in res["rows"][0]
        assert all("full_step_peak_reserved_bytes" in x for x in res["rows"])   # None on CPU, bytes on CUDA
        ld = json.loads((out / f"load-{cfg}.json").read_text())
        assert ld["all_loaded_diffs_zero"] and ld["keep_tag"] == S.KEEP_TAG
        assert (ld["counts"]["partly loaded"]["tensors"] == 0) == (cfg == "none")
        sn = json.loads((out / f"sanity-{cfg}.json").read_text())
        assert "predict_zero" in sn["loss_at_start"]
    assert not list(state.glob("*.pt")), "saved states are removed at the end"
    r = run_launch(feats, wan, out, state, configs="none")
    assert r.returncode != 0 and "exists" in r.stderr, "an existing measure file must not be overwritten"

    (wan / SHARD1).unlink()
    out2 = Path(root) / "results-broken"
    r = run_launch(feats, wan, out2, state, configs="none")
    res = json.loads((out2 / "measure-none.json").read_text())
    assert len(res["rows"]) == 1 and res["rows"][0]["outcome"] == "error" and res["rows"][0]["error_type"] == "LoadError"
    assert res["sweep_stopped"]["meaning"] == "error, not a memory limit" and res["optimizer_steps"] == []
    print("9 ok: launcher end to end on CPU (two configs, sweep 1-2, optimizer steps at m*), broken load stops")


MODAL_APP = Path(__file__).resolve().parents[1] / "experiments" / "2026-11-12-pretrained-start-sizing" / "modal_app.py"


def check_modal_app_container_import():
    """10. modal_app.py's module-level code runs as it would inside the Modal container: the file mounted at
    /root/modal_app.py (no repo above it) and modal.is_local() False. Paths resolve to the image's /root/src and no
    local file is read. Needs the modal package (1.6.4); without it, only the path function is checked."""
    import ast
    src = MODAL_APP.read_text()
    try:
        import modal
    except ImportError:
        modal = None
    if modal is not None:
        is_local = modal.is_local
        modal.is_local = lambda: False
        try:
            g = {"__file__": "/root/modal_app.py", "__name__": "modal_app"}
            exec(compile(src, "/root/modal_app.py", "exec"), g)
        finally:
            modal.is_local = is_local
        assert g["SRC"] == Path("/root/src") and g["REQS"] is None and g["TORCH_PINS"] == []
        how = "whole module executed (modal installed)"
    else:
        fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_paths")
        g = {"Path": Path, "CONTAINER_SRC": Path("/root/src")}
        exec(compile(ast.Module([fn], []), "modal_app.py", "exec"), g)
        assert g["_paths"](Path("/root"), False) == (Path("/root/src"), None)
        how = "_paths only (modal not installed)"
    try:
        Path("/root").parents[1]
        raise AssertionError("expected the old HERE.parents[1] to fail at /root")
    except IndexError:
        pass
    print(f"10 ok: modal_app imports as in the container ({how})")


def main():
    torch.manual_seed(0)
    check_modal_app_container_import()
    check_full_size_names()
    check_classify()
    check_m_star()
    with tempfile.TemporaryDirectory() as root:
        d = Path(root) / "wan-a"
        check_load(d)
        check_widened_equivalence(d)
        check_other_keep(Path(root) / "wan-b")
        check_load_errors(root)
        feats = Path(root) / "feats"
        fake_cache(feats, ["suite_a", "suite_b"], ["dinov3"], np.random.default_rng(0), dims={"dinov3": EXTRA})
        check_sanity(feats)
        check_launcher(root, feats)
    print("all checks passed")


def test_all():
    main()


if __name__ == "__main__":
    main()
