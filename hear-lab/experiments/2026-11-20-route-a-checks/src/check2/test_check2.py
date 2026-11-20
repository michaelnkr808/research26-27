"""CPU-only tests for src/check2 (run on the Mac and again on the machine).

    cd src/check2 && /opt/homebrew/bin/python3.11 -m unittest -v test_check2

What these cover: the start-up behaviour of sitecustomize.py (no effect without
LOCAL_RANK, no torch/transformers import at start-up, the lazy hook on a fake
transformers package), the step logger with a fake torch.cuda, the summariser, the
dry-run parsing and the shard/key bookkeeping of the key check on a tiny torch module.
If torch, transformers and accelerate are installed (on the machine; the Mac lacks
accelerate), one more test runs a tiny real Trainer on the
CPU through the hook (fake torch.cuda) to check that on_step_end comes once per
optimizer step with gradient accumulation. Not testable here: real CUDA numbers, the
WNM-3D model build and its 27 GB checkpoint, DeepSpeed, and the transformers 4.51.3 /
torch 2.8.0 versions of the machine (the Mac has other versions; the tests are rerun on
the machine, RUN.md step 6).
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import check2_keycheck as kc  # noqa: E402
import check2_steplog as sl  # noqa: E402

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_ST = importlib.util.find_spec("safetensors") is not None
HAVE_TF = importlib.util.find_spec("transformers") is not None
HAVE_ACC = importlib.util.find_spec("accelerate") is not None  # the real Trainer needs it


def run_py(code, env_extra=None, drop=("LOCAL_RANK", "CHECK2_HOOK_DISABLE"), extra_path=()):
    env = {k: v for k, v in os.environ.items() if k not in drop and k != "PYTHONPATH"}
    env["PYTHONPATH"] = os.pathsep.join([HERE, *extra_path])
    env.update(env_extra or {})
    out = subprocess.run(
        [sys.executable, "-s", "-c", code], env=env, capture_output=True, text=True, timeout=600
    )
    if out.returncode != 0:
        raise AssertionError(f"subprocess failed:\n{out.stdout}\n{out.stderr}")
    return out


PROBE = textwrap.dedent(
    """
    import json, sys, sitecustomize
    print(json.dumps({
        "file": sitecustomize.__file__,
        "finder": sitecustomize._CHECK2_FINDER is not None,
        "finder_in_meta_path": any(type(f).__name__ == "_Finder" for f in sys.meta_path),
        "torch": "torch" in sys.modules,
        "transformers": "transformers" in sys.modules,
    }))
    """
)


class TestHookStartup(unittest.TestCase):
    def probe(self, env_extra=None):
        out = run_py(PROBE, env_extra)
        return json.loads(out.stdout.strip().splitlines()[-1])

    def test_no_local_rank_does_nothing(self):
        r = self.probe()
        self.assertEqual(os.path.realpath(r["file"]), os.path.realpath(os.path.join(HERE, "sitecustomize.py")))
        self.assertFalse(r["finder"])
        self.assertFalse(r["finder_in_meta_path"])
        self.assertFalse(r["torch"])
        self.assertFalse(r["transformers"])

    def test_local_rank_installs_finder_without_importing(self):
        r = self.probe({"LOCAL_RANK": "0"})
        self.assertTrue(r["finder"])
        self.assertTrue(r["finder_in_meta_path"])
        self.assertFalse(r["torch"])
        self.assertFalse(r["transformers"])

    def test_disable_switch(self):
        r = self.probe({"LOCAL_RANK": "0", "CHECK2_HOOK_DISABLE": "1"})
        self.assertFalse(r["finder"])
        self.assertFalse(r["finder_in_meta_path"])


FAKE_TF_INIT = "from .trainer_callback import TrainerCallback\n"
FAKE_TF_CB = "class TrainerCallback:\n    pass\n"
FAKE_TF_TRAINER = textwrap.dedent(
    """
    class _Handler:
        def __init__(self):
            self.callbacks = []

    class Trainer:
        def __init__(self, name="t"):
            self.name = name
            self.callback_handler = _Handler()

        def add_callback(self, cb):
            self.callback_handler.callbacks.append(cb)
    """
)

FAKE_USE = textwrap.dedent(
    """
    import json, sys
    import transformers.trainer as tt
    from check2_steplog import Check2StepLogger

    class Sub(tt.Trainer):
        def __init__(self):
            super().__init__(name="sub")

    a, b, c = tt.Trainer(), tt.Trainer(), Sub()
    count = lambda t: sum(isinstance(x, Check2StepLogger) for x in t.callback_handler.callbacks)
    print(json.dumps({"a": count(a), "b": count(b), "c": count(c),
                      "patched": getattr(tt.Trainer.__init__, "_check2_patched", False),
                      "torch": "torch" in sys.modules}))
    """
)


class TestHookOnFakeTransformers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        pkg = os.path.join(self.tmp.name, "transformers")
        os.makedirs(pkg)
        for name, body in (("__init__.py", FAKE_TF_INIT), ("trainer_callback.py", FAKE_TF_CB), ("trainer.py", FAKE_TF_TRAINER)):
            with open(os.path.join(pkg, name), "w") as fh:
                fh.write(body)

    def tearDown(self):
        self.tmp.cleanup()

    def run_fake(self, env_extra):
        out = run_py(FAKE_USE, env_extra, extra_path=(self.tmp.name,))
        return json.loads(out.stdout.strip().splitlines()[-1]), out.stderr

    def test_attaches_once_per_trainer_with_local_rank(self):
        r, err = self.run_fake({"LOCAL_RANK": "0"})
        self.assertEqual((r["a"], r["b"], r["c"]), (1, 1, 1))
        self.assertTrue(r["patched"])
        self.assertFalse(r["torch"])  # attaching does not import torch either
        self.assertIn("[check2] step logger attached", err)

    def test_no_attach_without_local_rank(self):
        r, err = self.run_fake({})
        self.assertEqual((r["a"], r["b"], r["c"]), (0, 0, 0))
        self.assertFalse(r["patched"])
        self.assertNotIn("[check2]", err)



@unittest.skipUnless(HAVE_TF and HAVE_TORCH, "needs transformers and torch")
class TestHookOnRealTransformersImport(unittest.TestCase):
    """The real transformers package loads its trainer lazily (importlib.import_module)."""

    def test_lazy_import_gets_patched(self):
        code = (
            "import json, sys\n"
            "from transformers import Trainer\n"
            "print(json.dumps({'patched': getattr(Trainer.__init__, '_check2_patched', False),"
            " 'mod': 'transformers.trainer' in sys.modules}))\n"
        )
        out = run_py(code, {"LOCAL_RANK": "0", "USE_TF": "0"})
        r = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertTrue(r["mod"])
        self.assertTrue(r["patched"])


class FakeCuda:
    def __init__(self, available=True):
        self.available = available
        self.calls = []
        self.alloc = 100

    def is_available(self):
        return self.available

    def synchronize(self):
        self.calls.append("synchronize")

    def memory_allocated(self):
        self.calls.append("memory_allocated")
        return self.alloc

    def max_memory_allocated(self):
        self.calls.append("max_memory_allocated")
        return self.alloc * 2

    def memory_reserved(self):
        self.calls.append("memory_reserved")
        return self.alloc * 3

    def max_memory_reserved(self):
        self.calls.append("max_memory_reserved")
        return self.alloc * 4

    def reset_peak_memory_stats(self):  # must never be called
        raise AssertionError("peak counters must not be reset")


class TestStepLogger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.args = types.SimpleNamespace(output_dir=self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_records_and_order(self):
        cuda = FakeCuda()
        ticks = iter([10.0, 12.5, 15.0, 16.0])
        cb = sl.Check2StepLogger(cuda=cuda, clock=lambda: next(ticks))
        st = types.SimpleNamespace(global_step=0, is_world_process_zero=True)
        cb.on_train_begin(self.args, st, None)
        for k in (1, 2):
            st.global_step = k
            cuda.alloc += 50
            cb.on_step_end(self.args, st, None)
        cb.on_train_end(self.args, st, None)
        self.assertEqual(cuda.calls[0], "synchronize")
        self.assertEqual(cuda.calls.count("synchronize"), 4)
        recs = sl.read_jsonl(os.path.join(self.tmp.name, sl.STEPS_FILENAME))
        self.assertEqual([r["event"] for r in recs], ["train_begin", "step_end", "step_end", "train_end"])
        self.assertEqual([r["global_step"] for r in recs], [0, 1, 2, 2])
        self.assertEqual([r["time"] for r in recs], [10.0, 12.5, 15.0, 16.0])
        for r in recs:
            for f in sl.MEMORY_FIELDS:
                self.assertIsInstance(r[f], int)
        self.assertEqual(recs[1]["max_memory_reserved"], 600)

    def test_sync_before_every_read(self):
        cuda = FakeCuda()
        cb = sl.Check2StepLogger(cuda=cuda, clock=lambda: 1.0)
        cb.on_step_end(self.args, types.SimpleNamespace(global_step=1), None)
        self.assertEqual(cuda.calls, ["synchronize", "memory_allocated", "max_memory_allocated", "memory_reserved", "max_memory_reserved"])

    def test_not_world_zero_writes_nothing(self):
        cb = sl.Check2StepLogger(cuda=FakeCuda(), clock=lambda: 1.0)
        cb.on_step_end(self.args, types.SimpleNamespace(global_step=1, is_world_process_zero=False), None)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, sl.STEPS_FILENAME)))

    def test_no_cuda_raises(self):
        cb = sl.Check2StepLogger(cuda=FakeCuda(available=False))
        with self.assertRaises(RuntimeError):
            cb.on_train_begin(self.args, types.SimpleNamespace(global_step=0), None)


def rec(event, step, t, alloc_peak, res_peak):
    return {"event": event, "global_step": step, "time": t, "memory_allocated": 1,
            "max_memory_allocated": alloc_peak, "memory_reserved": 2, "max_memory_reserved": res_peak}


class TestSummarize(unittest.TestCase):
    def test_durations_and_peaks(self):
        recs = [rec("train_begin", 0, 100.0, 50, 60)]
        t = 100.0
        for k, d in enumerate([9.0, 3.0, 2.0, 2.0, 4.0], start=1):
            t += d
            recs.append(rec("step_end", k, t, 50 if k < 3 else 70, 80))
        recs.append(rec("train_end", 5, t + 1, 70, 80))
        s = sl.summarize(recs, warmup=2, batch=12)
        self.assertEqual(s["step_seconds_all"], [9.0, 3.0, 2.0, 2.0, 4.0])
        self.assertEqual(s["timed_steps"], 3)
        self.assertEqual(s["timed_median_s"], 2.0)
        self.assertAlmostEqual(s["timed_mean_s"], 8.0 / 3)
        self.assertEqual((s["timed_min_s"], s["timed_max_s"]), (2.0, 4.0))
        self.assertEqual(s["samples_per_s_at_median"], 6.0)
        self.assertEqual(s["setup_peak"]["max_memory_allocated"], 50)
        self.assertEqual(s["final_peak"]["max_memory_allocated"], 70)
        self.assertFalse(s["peak_allocated_from_setup"])
        self.assertFalse(s["peak_reserved_from_setup"])

    def test_peak_from_setup_flag(self):
        recs = [rec("train_begin", 0, 0.0, 90, 95), rec("step_end", 1, 1.0, 90, 95)]
        s = sl.summarize(recs, warmup=0)
        self.assertTrue(s["peak_allocated_from_setup"])
        self.assertTrue(s["peak_reserved_from_setup"])

    def test_gap_in_steps_rejected(self):
        recs = [rec("train_begin", 0, 0.0, 1, 1), rec("step_end", 1, 1.0, 1, 1), rec("step_end", 3, 2.0, 1, 1)]
        with self.assertRaises(ValueError):
            sl.summarize(recs, warmup=0)


DRY_RUN_TEXT = textwrap.dedent(
    r"""
    Stage: stage2
    Dataset: data/interiorgs_lerobot_check2
    Batch: per_device=12 global=12
    Command: python3 -m torch.distributed.run --nproc_per_node 1 --standalone /workspace/WNM-3D/gammanav/vln/experiment/experiment.py report_to=tensorboard data=wnm_3d/interiorgs_wan22 num_frames=33 training_args.deepspeed=gammanav/vln/configs/deepspeed/zero2.json +training_args.logging_dir=/workspace/check2/runs/t1/tensorboard output_dir=/workspace/check2/runs/t1 per_device_train_batch_size=12 global_batch_size=12 pretrained_model_path=checkpoints/wnm_3d_stage3_release ++action_head_cfg.config.skip_component_loading=true path\ with\ space=1
    """
)


class TestKeycheckParsing(unittest.TestCase):
    def test_overrides(self):
        ov = kc.overrides_from_dry_run(DRY_RUN_TEXT)
        self.assertEqual(ov[0], "report_to=tensorboard")
        self.assertIn("++action_head_cfg.config.skip_component_loading=true", ov)
        self.assertEqual(ov[-1], "path with space=1")
        self.assertNotIn("--standalone", ov)
        self.assertEqual(kc.pretrained_path(ov), "checkpoints/wnm_3d_stage3_release")

    def test_rejects_missing_command_or_checkpoint(self):
        with self.assertRaises(ValueError):
            kc.overrides_from_dry_run("Stage: x\n")
        with self.assertRaises(ValueError):
            kc.overrides_from_dry_run("Command: python3 /a/experiment/experiment.py seed=1\n")


@unittest.skipUnless(HAVE_TORCH and HAVE_ST, "needs torch and safetensors")
class TestKeycheckKeys(unittest.TestCase):
    def setUp(self):
        import torch
        from safetensors.torch import save_file

        self.torch = torch
        self.save_file = save_file
        self.tmp = tempfile.TemporaryDirectory()

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Linear(2, 2)
                self.b = torch.nn.Linear(2, 3)

        self.Tiny = Tiny

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, shards):
        weight_map = {}
        for name, sd in shards.items():
            self.save_file(sd, os.path.join(self.tmp.name, name))
            weight_map.update({k: name for k in sd})
        with open(os.path.join(self.tmp.name, "model.safetensors.index.json"), "w") as fh:
            json.dump({"metadata": {}, "weight_map": weight_map}, fh)

    def test_full_match_passes_and_loads_values(self):
        from safetensors.torch import load_file

        src = self.Tiny()
        sd = {k: v.clone() + 1.0 for k, v in src.state_dict().items()}
        self.write({"model-00002.safetensors": {k: sd[k] for k in ("b.weight", "b.bias")},
                    "model-00001.safetensors": {k: sd[k] for k in ("a.weight", "a.bias")}})
        model = self.Tiny()
        r = kc.check_keys(model, self.tmp.name, load_file, log=lambda *_: None)
        self.assertTrue(r["pass"])
        self.assertEqual((r["n_missing"], r["n_unexpected"], r["n_shards"]), (0, 0, 2))
        self.assertTrue(r["index_matches_shards"])
        self.assertEqual([p["shard"] for p in r["per_shard"]], ["model-00001.safetensors", "model-00002.safetensors"])
        self.assertTrue(self.torch.equal(model.a.weight, sd["a.weight"]))

    def test_missing_and_unexpected_reported(self):
        from safetensors.torch import load_file

        sd = self.Tiny().state_dict()
        self.write({"model-00001.safetensors": {"a.weight": sd["a.weight"].clone(), "a.bias": sd["a.bias"].clone(),
                                                "extra.weight": self.torch.zeros(1)}})
        r = kc.check_keys(self.Tiny(), self.tmp.name, load_file, log=lambda *_: None)
        self.assertFalse(r["pass"])
        self.assertEqual(r["missing_keys"], ["b.bias", "b.weight"])
        self.assertEqual(r["unexpected_keys"], ["extra.weight"])

    def test_shape_mismatch_raises(self):
        from safetensors.torch import load_file

        self.write({"model-00001.safetensors": {"a.weight": self.torch.zeros(5, 5)}})
        with self.assertRaises(RuntimeError):
            kc.check_keys(self.Tiny(), self.tmp.name, load_file, log=lambda *_: None)


REAL_TRAINER = textwrap.dedent(
    """
    import json, os, sys
    import check2_steplog

    class FakeCuda:
        def is_available(self): return True
        def synchronize(self): pass
        def memory_allocated(self): return 1
        def max_memory_allocated(self): return 2
        def memory_reserved(self): return 3
        def max_memory_reserved(self): return 4

    check2_steplog._default_cuda = lambda: FakeCuda()   # test only: no GPU on the Mac
    import torch
    from transformers import Trainer, TrainingArguments

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(4, 1)
        def forward(self, x, y):
            return {"loss": ((self.l(x) - y) ** 2).mean()}

    class DS(torch.utils.data.Dataset):
        def __len__(self): return 64
        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return {"x": torch.randn(4, generator=g), "y": torch.randn(1, generator=g)}

    out = sys.argv[1]
    args = TrainingArguments(output_dir=out, max_steps=3, per_device_train_batch_size=2,
                             gradient_accumulation_steps=int(sys.argv[2]), use_cpu=True,
                             report_to=[], save_strategy="no", logging_steps=10,
                             dataloader_num_workers=0, disable_tqdm=True)
    Trainer(model=M(), args=args, train_dataset=DS()).train()
    rows = [json.loads(l) for l in open(os.path.join(out, "check2_steps.jsonl"))]
    print(json.dumps([[r["event"], r["global_step"]] for r in rows]))
    """
)


@unittest.skipUnless(HAVE_TORCH and HAVE_TF and HAVE_ACC, "needs torch, transformers and accelerate (runs on the machine)")
class TestRealTrainerCPU(unittest.TestCase):
    def run_trainer(self, accum):
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "t.py")
            with open(script, "w") as fh:
                fh.write(REAL_TRAINER)
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            # USE_TF=0: the Mac has a broken TensorFlow install that transformers would import
            env.update(PYTHONPATH=HERE, USE_TF="0", LOCAL_RANK="0", RANK="0", WORLD_SIZE="1",
                       MASTER_ADDR="127.0.0.1", MASTER_PORT=str(29500 + os.getpid() % 1000 + accum))
            out = subprocess.run([sys.executable, "-s", script, os.path.join(tmp, "out"), str(accum)],
                                 env=env, capture_output=True, text=True, timeout=600)
            if out.returncode != 0:
                raise AssertionError(out.stdout + out.stderr)
            self.assertIn("[check2] step logger attached", out.stderr)
            return json.loads(out.stdout.strip().splitlines()[-1])

    def test_once_per_optimizer_step(self):
        expected = [["train_begin", 0], ["step_end", 1], ["step_end", 2], ["step_end", 3], ["train_end", 3]]
        self.assertEqual(self.run_trainer(1), expected)
        self.assertEqual(self.run_trainer(2), expected)


if __name__ == "__main__":
    unittest.main()
