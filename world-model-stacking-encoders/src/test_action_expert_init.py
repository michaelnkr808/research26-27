"""CPU checks for action_expert_init.py. Run from src/: python test_action_expert_init.py

Needs FastWAM 7faa711 importable (pip install --no-deps -e FastWAM), omegaconf and safetensors, and the Fast-WAM source
tree (for scripts/preprocess_action_dit_backbone.py, which is not part of the installed package): FASTWAM_TREE, or the
tree the installed `fastwam` package lives in (<tree>/src/fastwam).
"""

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import torch

import action_expert_init as A
import pretrained_sizing as S
import proxy_wam as P
from test_pretrained_sizing import TINY, fake_wan


def fastwam_tree():
    if os.environ.get("FASTWAM_TREE"):
        return Path(os.environ["FASTWAM_TREE"])
    import fastwam
    return Path(fastwam.__file__).resolve().parents[2]


def load_script():
    """Fast-WAM's own preprocess script, imported by file path (test-only dependency)."""
    path = fastwam_tree() / "scripts" / "preprocess_action_dit_backbone.py"
    assert path.exists(), f"{path} missing: set FASTWAM_TREE to the Fast-WAM 7faa711 source tree"
    spec = importlib.util.spec_from_file_location("fw_preprocess_action_dit_backbone", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def experts_on_meta(layers, model_kw):
    """(video expert, action expert) exactly as proxy_wam.build_model builds them, on the meta device (no memory).
    FastWAMJoint itself can't be built on meta (its schedulers call .item()), so it is stubbed for this call only."""
    orig = P.FastWAMJoint
    P.FastWAMJoint = lambda video, action, mot, **kw: types.SimpleNamespace(video_expert=video, action_expert=action)
    try:
        with torch.device("meta"):
            m = P.build_model(P.VAE_DIM, "meta", layers=layers, **model_kw)
    finally:
        P.FastWAMJoint = orig
    return m.video_expert, m.action_expert


def run_fastwam_script(fw, video_expert, action_kw, out_path):
    """Run Fast-WAM's script main() end to end on our video expert: its Wan loader is replaced by one that returns
    `video_expert` (the only change; the yaml, the ActionDiT build, the key loop, resize, alpha and save are its own)."""
    yaml = out_path.with_suffix(".yaml")
    lines = ["video_dit_config:", "  num_layers: %d" % len(video_expert.blocks), "action_dit_config:"]
    lines += [f"  {k}: {v}" for k, v in action_kw.items()]
    yaml.write_text("\n".join(lines) + "\n")
    fw.load_wan22_ti2v_5b_components = lambda **kw: types.SimpleNamespace(dit=video_expert)
    argv = sys.argv
    sys.argv = ["preprocess_action_dit_backbone.py", "--model-config", str(yaml), "--output", str(out_path),
                "--device", "cpu", "--dtype", "float32"]
    try:
        fw.main()
    finally:
        sys.argv = argv
    return torch.load(out_path, map_location="cpu", weights_only=False)


def main():
    fw = load_script()

    # 1. The copied helpers are Fast-WAM's: same outputs (torch.equal) on random tensors at real Wan sizes for each
    #    resize class (both dims, last dim only, first dim only, 3-D modulation).
    g = torch.Generator().manual_seed(0)
    for src, tgt in [((3072, 14336), (1024, 4096)), ((18432, 3072), (6144, 1024)), ((3072, 3072), (3072, 1024)),
                     ((3072, 3072), (1024, 3072)), ((1, 6, 3072), (1, 6, 1024)), ((14336,), (4096,))]:
        x = torch.randn(src, generator=g)
        assert torch.equal(A._resize_tensor_to_shape(x, tgt), fw._resize_tensor_to_shape(x, tgt)), (src, tgt)
    print("1 ok: resize helpers equal Fast-WAM's on real Wan shapes")

    # 2. End to end on a tiny fake Wan checkpoint: build, load_wan (blocks 0-1 of 4), derive. The derived action
    #    backbone equals (torch.equal) the payload Fast-WAM's own script writes from the same video expert; the fresh
    #    tensors (action_encoder, head) and the proprio encoder keep their build init; the video expert is unchanged.
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fake_wan(d / "wan", n_blocks=4)
        model = S.build(P.VAE_DIM, "cpu", 2, TINY, seed=0)
        S.load_wan(model, d / "wan", [0, 1], log=lambda m: None)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rep = A.init_action_expert(model, log=lambda m: None)
        after = model.state_dict()
        a = model.action_expert
        action_kw = {"hidden_dim": a.hidden_dim, "action_dim": 7, "ffn_dim": a.blocks[0].ffn_dim,
                     "text_dim": P.TEXT_DIM, "freq_dim": P.FREQ_DIM, "eps": 1e-6, "num_heads": a.num_heads,
                     "attn_head_dim": a.attn_head_dim, "num_layers": len(a.blocks)}
        payload = run_fastwam_script(fw, model.video_expert, action_kw, d / "backbone.pt")
        ref = payload["backbone_state_dict"]
        mine = {k[len("action_expert."):]: v for k, v in after.items() if k.startswith("action_expert.")}
        backbone = {k for k in mine if not k.startswith(A.SKIP_PREFIXES)}
        assert set(ref) == backbone, (set(ref) ^ backbone)
        assert all(torch.equal(mine[k], ref[k]) for k in ref), [k for k in ref if not torch.equal(mine[k], ref[k])]
        for k, v in after.items():
            if k.startswith(("video_expert.", "proprio_encoder.")) or k[len("action_expert."):].startswith(A.SKIP_PREFIXES):
                assert torch.equal(v, before[k]), k
        changed = sum(not torch.equal(after["action_expert." + k], before["action_expert." + k]) for k in ref)
        assert changed == len(ref), (changed, len(ref))
        c = rep["counts"]
        assert c["derived"] == len(ref) and c["fresh"] == 4 and c["copied"] > 0 and c["resized"] > 0, c
    print(f"2 ok: derived action expert equals Fast-WAM's script output bit for bit ({len(ref)} tensors, "
          f"{c['copied']} copied, {c['resized']} resized, 4 fresh)")

    # 3. At the real sizes (Wan 2.2 TI2V-5B width, 8 layers, shapes only on the meta device): 226 derived (80 copied,
    #    146 resized) + 4 fresh; with the proprio encoder's 2 tensors the fresh total of the sizing run (232) is matched
    #    by derived + fresh + proprio; alpha per class as fastwam-reading.md Q5.
    video, action = experts_on_meta(8, S.WAN_KW)
    vs = {k: v.shape for k, v in video.state_dict().items()}
    acs = {k: v.shape for k, v in action.state_dict().items()}
    kinds = A.classify(vs, acs)
    n = {c: sum(1 for x in kinds.values() if x == c) for c in ("copied", "resized", "fresh")}
    assert n == {"copied": 80, "resized": 146, "fresh": 4}, n
    proprio = 2   # proprio_encoder weight and bias (Linear(8, 4096)), outside the action expert
    assert n["copied"] + n["resized"] + n["fresh"] + proprio == 232
    al = A.alpha_values(kinds, vs, acs)
    s3, s35 = 3 ** 0.5, 3.5 ** 0.5
    want = {"blocks.0.self_attn.q.weight": s3, "blocks.0.modulation": s3, "text_embedding.2.weight": s3,
            "time_projection.1.weight": s3, "blocks.0.ffn.0.weight": s3, "blocks.0.ffn.2.weight": s35,
            "blocks.0.self_attn.o.weight": None, "text_embedding.0.weight": None, "time_projection.1.bias": None}
    for k, v in want.items():
        assert (al[k] is None and v is None) or abs(al[k] - v) < 1e-12, (k, al[k], v)
    print("3 ok: real sizes give 226 derived (80 copied, 146 resized) + 4 fresh; + 2 proprio = 232; alpha per class")
    print("all checks passed")


if __name__ == "__main__":
    main()
