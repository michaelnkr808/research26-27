#!/usr/bin/env python3
"""pfb-jobs done-check helper: parse every job-pfb-*.yaml (nine expected), extract the container script, run bash -n, check the resource, policy and phase-B rules.
Usage: check-pfb-jobs.py <jobs dir> <scratch dir> [<src dir>]    (src dir default: ../../../src of this file; used to cross-check the CLI flags the scripts pass)
Started from check-pfa-jobs.py (same checks), plus the phase-B specifics: pinned digest in every Job and IMAGE_DIGEST equal to it in the render Job; per-condition staging paths;
--consistency-ref on every test render and --expect-draws on every perturbed one; --config /pf/phase-a.yaml everywhere; revision asserts; no sleep/while/until; CPU Jobs without GPU."""
import glob
import re
import subprocess
import sys
from pathlib import Path

import yaml

jobs = Path(sys.argv[1])
scratch = Path(sys.argv[2])
scratch.mkdir(parents=True, exist_ok=True)
src_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).resolve().parents[3] / "src"
DIGEST = "pytorch/pytorch@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2"
CONFIGMAP = "stacking-encoders-pf-main-src-b"
CONDS = ["clean", "camera", "light", "background", "noise"]
EXPECTED = (["job-pfb-draws.yaml", "job-pfb-render.yaml", "job-pfb-ext-vggt.yaml"] + [f"job-pfb-ext-sam3-{c}.yaml" for c in CONDS] + ["job-pfb-final.yaml"])
CPU_ONLY = {"job-pfb-draws.yaml", "job-pfb-final.yaml"}
SIGLIP2_SHA = "1b426889ea62b5a72bf9839009a1b184bfc9c178"     # results/pf-main/phase-a/pfa-ext-siglip2-cradio/revisions.txt
CONFIG_ARG = "/pf/phase-a.yaml"


def num(v):
    v = str(v)
    m = re.fullmatch(r"([0-9.]+)(m|Ki|Mi|Gi)?", v)
    x, u = float(m.group(1)), m.group(2)
    return x / 1000 if u == "m" else x * {"Ki": 1 / 1024, "Mi": 1, "Gi": 1024}[u] if u else x


def logical_lines(script):
    """Script lines with backslash continuations joined."""
    out, cur = [], ""
    for ln in script.split("\n"):
        if ln.rstrip().endswith("\\"):
            cur += ln.rstrip()[:-1] + " "
        else:
            out.append(cur + ln)
            cur = ""
    return out


def source_flags(path):
    """Flags (--x) named in add_argument calls of a source file (the fallback when --help cannot be run)."""
    return set(re.findall(r"add_argument\(\s*(?:\"[^\"]*\",\s*)*\"(--[a-z0-9-]+)\"", Path(path).read_text())) | \
        set(re.findall(r"\"(--[a-z0-9-]+)\"", "\n".join(ln for ln in Path(path).read_text().split("\n") if "add_argument" in ln)))


FLAGS = {}
if src_dir.is_dir():
    for tool in ("extract_features.py", "pf_rerender.py", "pf_analysis.py"):
        if (src_dir / tool).is_file():
            FLAGS[tool] = source_flags(src_dir / tool)

bad = 0
missing = [n for n in EXPECTED if not (jobs / n).is_file()]
extra = sorted(Path(f).name for f in glob.glob(str(jobs / "job-pfb-*.yaml")) if Path(f).name not in EXPECTED)
if missing or extra:
    print("PROBLEMS: expected manifests missing", missing, "unexpected", extra)
    bad += 1
names = set()
total_gpu = 0
for f in sorted(glob.glob(str(jobs / "job-pfb-*.yaml"))):
    name = Path(f).name
    docs = list(yaml.safe_load_all(open(f)))
    msgs = []
    if len(docs) != 1:
        print(f"{name}: {len(docs)} documents")
        bad += 1
        continue
    d = docs[0]
    spec = d["spec"]
    pod = spec["template"]["spec"]
    c = pod["containers"][0]
    jname = d["metadata"]["name"]
    if not jname.startswith("stacking-encoders-pfb-") or jname in names:
        msgs.append(f"job name {jname!r} not unique or not stacking-encoders-pfb-*")
    names.add(jname)
    if d["metadata"].get("namespace") != "aria-world":
        msgs.append("namespace")
    if spec.get("backoffLimit") != 0:
        msgs.append("backoffLimit != 0")
    if pod.get("restartPolicy") != "Never":
        msgs.append("restartPolicy")
    if "tolerations" in pod:
        msgs.append("tolerations present")
    if c["image"] != DIGEST:
        msgs.append("image not the pinned digest")
    envd = {e["name"]: e.get("value") for e in c["env"]}
    is_render = name == "job-pfb-render.yaml"
    if is_render and envd.get("IMAGE_DIGEST") != c["image"]:
        msgs.append("IMAGE_DIGEST env differs from the image")
    if is_render and envd.get("IMAGE_DIGEST") != DIGEST:
        msgs.append("IMAGE_DIGEST is not the pinned digest")
    if envd.get("SRC_GIT_COMMIT") != "FILL_AT_SUBMIT":
        msgs.append("SRC_GIT_COMMIT env is not FILL_AT_SUBMIT")
    if envd.get("PYTHONPATH") != "/pf:/src":
        msgs.append(f"PYTHONPATH env {envd.get('PYTHONPATH')!r}")
    req, lim = c["resources"]["requests"], c["resources"]["limits"]
    cpu_only = name in CPU_ONLY
    ngpu = int(req.get("nvidia.com/gpu", 0))
    if ngpu != (0 if cpu_only else 1) or int(lim.get("nvidia.com/gpu", 0)) != ngpu:
        msgs.append(f"gpu request {ngpu}")
    total_gpu += ngpu
    for k in ("cpu", "memory", "ephemeral-storage"):
        r, l = num(req[k]), num(lim[k])
        if not (r <= l <= 1.2 * r + 1e-9):
            msgs.append(f"{k}: request {req[k]} limit {lim[k]} ratio {l / r:.3f}")
    vols = {v["name"]: v for v in pod["volumes"]}
    if not any(n == "dshm" for n in vols) and not cpu_only:
        msgs.append("no /dev/shm emptyDir")
    if vols.get("pf", {}).get("configMap", {}).get("name") != CONFIGMAP:
        msgs.append(f"/pf is not the ConfigMap {CONFIGMAP}")
    if vols.get("src", {}).get("configMap", {}).get("name") != "stacking-encoders-proxy-src":
        msgs.append("/src is not stacking-encoders-proxy-src")
    aff = str(pod["affinity"])
    if "us-west" not in aff:
        msgs.append("no us-west")
    if not cpu_only and "ry-gpu-08" not in aff:
        msgs.append("excluded hosts missing")
    if cpu_only and "pci-10de" not in aff:
        msgs.append("no GPU-node anti-affinity")
    if cpu_only and "nvidia.com/gpu.product" in aff:
        msgs.append("CPU job selects a GPU product")
    if is_render and "['NVIDIA-L4']" not in aff:
        msgs.append("render not L4 only")
    if spec.get("ttlSecondsAfterFinished") is None or not spec.get("activeDeadlineSeconds"):
        msgs.append("ttl or deadline missing")
    script = c["args"][0]
    sp = scratch / (Path(f).stem + ".sh")
    sp.write_text(script)
    r = subprocess.run(["bash", "-n", str(sp)], capture_output=True, text=True)
    if r.returncode != 0 or r.stderr.strip():
        msgs.append("bash -n: " + r.stderr.strip()[:300])
    if re.search(r"\bsleep\b", script):
        msgs.append("sleep in script")
    if re.search(r"\bwhile\b|\buntil\b", script):
        msgs.append("loop keyword while/until in script")
    if "tolerations" in script:
        msgs.append("'tolerations' in script")
    if 'case "${SRC_GIT_COMMIT:-}" in FILL*' not in script:
        msgs.append("no SRC_GIT_COMMIT guard")
    if "never overwritten" not in script and "never overwrites" not in script:
        msgs.append("no never-overwrite guard")
    if 'lt 100' not in script:
        msgs.append("no 100 GB free-space abort")
    if "timeout -k 60" not in script:
        msgs.append("no stage cap (timeout)")
    # heredocs must be closed at column 0
    for m in re.finditer(r"<<'(\w+)'", script):
        if not re.search(r"^" + m.group(1) + r"$", script, re.M):
            msgs.append(f"heredoc {m.group(1)} has no terminator at column 0")
    lines = logical_lines(script)
    code = "\n".join(ln for ln in script.split("\n") if not ln.strip().startswith("#") and not re.search(r"\s#\s", ln))   # without comment lines and trailing comments
    # --config: always /pf/phase-a.yaml
    cfgs = set(re.findall(r"--config\s+(\S+)", script))
    if cfgs and cfgs != {CONFIG_ARG}:
        msgs.append(f"--config values {sorted(cfgs)}")
    for ln in lines:
        if "pf_rerender.py " in ln and "sha256sum" not in ln and not ln.strip().startswith("#"):
            if f"--config {CONFIG_ARG}" not in ln:
                msgs.append("pf_rerender line without --config /pf/phase-a.yaml: " + ln.strip()[:100])
    # --- per Job specifics
    rerender = [ln for ln in lines if "/pf/pf_rerender.py " in ln and "sha256sum" not in ln and not ln.strip().startswith("#")]
    if name == "job-pfb-draws.yaml":
        if len(rerender) != 1 or " draws " not in rerender[0] or "--out /work/draws/draws.json" not in rerender[0]:
            msgs.append("draws job must run exactly one pf_rerender.py draws --out /work/draws/draws.json")
        if "=== BEGIN DRAWS_JSON" not in script or "=== END DRAWS_JSON" not in script or "sha256sum /work/draws/draws.json" not in script:
            msgs.append("draws job: markers or sha256sum missing")
        if "env_bundle.sh restore" not in script:
            msgs.append("draws job: no env restore (the classification file comes from the bundle)")
    if is_render:
        seen = {}
        for ln in rerender:
            m = re.search(r" render .*--condition (\w+)", ln)
            if not m:
                msgs.append("pf_rerender line that is not a render: " + ln.strip()[:100])
                continue
            cond = m.group(1)
            seen[cond] = seen.get(cond, 0) + 1
            if "--split test" not in ln:
                msgs.append(f"render {cond}: not --split test")
            if "--consistency-ref $PA/render" not in ln:
                msgs.append(f"render {cond}: no --consistency-ref $PA/render")
            if cond != "clean" and "--expect-draws /pf/draws.json" not in ln:
                msgs.append(f"render {cond}: perturbed without --expect-draws /pf/draws.json")
            if cond == "clean" and "--expect-draws" in ln:
                msgs.append("render clean with --expect-draws")
            if f"--out /work/render/{cond} " not in ln + " ":
                msgs.append(f"render {cond}: --out is not /work/render/{cond}")
            if "--pilot" in ln or "--max-demos" in ln:
                msgs.append(f"render {cond}: pilot flags")
        if seen != {c_: 1 for c_ in CONDS}:
            msgs.append(f"render conditions {seen}")
        if "PA=$R/pf-main/phase-a" not in script or "R=/mnt/proxy" not in script:
            msgs.append("PA is not /mnt/proxy/pf-main/phase-a")
        if not re.search(r"^ *NOISE_CAP=FILL_AT_SUBMIT$", script, re.M):
            msgs.append("NOISE_CAP variable missing or not the FILL_AT_SUBMIT placeholder")
        if "case \"$NOISE_CAP\" in" not in script or "exit 11" not in script.split("case \"$NOISE_CAP\" in", 1)[-1][:300]:
            msgs.append("no NOISE_CAP abort guard (exit 11)")
        if not re.search(r"^ *CONDS=\"clean camera light background noise\"$", script, re.M):
            msgs.append("render CONDS default is not all five conditions")
        if "cap $NOISE_CAP python /pf/pf_rerender.py render" not in script:
            msgs.append("the noise render does not use cap $NOISE_CAP")
        if "ea8dc28" not in script or SIGLIP2_SHA not in script:
            msgs.append("DINOv3 / SigLIP 2 revision asserts missing")
        if "transformers==5.18.0" not in script or "transformers==4.57.6" not in script:
            msgs.append("transformers pins missing")
        if "/work/extlib" not in script or "/work/venv457" not in script:
            msgs.append("extraction environments missing")
        if "pip install" in script and re.search(r"^\s*cap \d+ pip install(?! --no-cache-dir -q --target)", script, re.M):
            msgs.append("a pip install into the simulator venv (must be --target)")
        for e in ("dinov3", "siglip2", "c-radiov4"):
            if f"--encoder {e}" not in script:
                msgs.append(f"no extraction of {e}")
    if name.startswith("job-pfb-ext-") or is_render:
        # per-condition staging paths
        toks = re.findall(r"/work/pfb-test-([^/\s\"']*)/", code)
        okt = set(CONDS) | {"$_c", "$c", "{cond}"}
        for t in toks:
            if t not in okt:
                msgs.append(f"staging path token {t!r}")
        if "/work/pfa-" in script:
            msgs.append("a phase-A staging path is used")
        if not toks:
            msgs.append("no per-condition staging path")
        if is_render and "/work/pfb-test-$_c/libero_goal/*.hdf5" not in script:
            msgs.append("render: extraction does not read /work/pfb-test-$_c/libero_goal/*.hdf5")
        if is_render and "--out /work/feats/$_c" not in script:
            msgs.append("render: extraction output is not per condition")
    if name.startswith("job-pfb-ext-"):
        m = re.search(r'^CONDS="([^"]*)"', script, re.M)
        conds = m.group(1).split() if m else None
        enc = re.search(r"^ENC=(\S+)", script, re.M)
        enc = enc.group(1) if enc else None
        if name == "job-pfb-ext-vggt.yaml":
            if conds != CONDS:
                msgs.append(f"vggt CONDS {conds}")
            if enc != "vggt-omega" or "1041e80" not in script or "facebook/VGGT-Omega" not in script:
                msgs.append("vggt: encoder or revision assert")
            if "transformers==5.18.0" not in script or "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6" not in script:
                msgs.append("vggt: pins")
        else:
            cond = name[len("job-pfb-ext-sam3-"):-len(".yaml")]
            if conds != [cond]:
                msgs.append(f"{name}: CONDS {conds}, expected [{cond}]")
            if enc != "sam3" or "3c879f3" not in script or "facebook/sam3" not in script:
                msgs.append("sam3: encoder or revision assert")
            if "transformers==5.18.0" not in script:
                msgs.append("sam3: pins")
        if conds and len(set(conds)) != len(conds):
            msgs.append("duplicate conditions")
        if len(rerender):
            msgs.append("an extraction Job runs pf_rerender")
        for cnd in conds or []:
            pass
        if '$PB/render/$c/RENDER_COMPLETE' not in script:
            msgs.append("no RENDER_COMPLETE gate per condition")
        if "--out /work/feats/$c" not in script or "/work/pfb-test-$c/libero_goal/*.hdf5" not in script:
            msgs.append("extraction not per condition")
        if '[ $LAST -eq 0 ]; then rm -rf' not in script:
            msgs.append("features not freed after the copy")
    if name == "job-pfb-final.yaml":
        fl = [ln for ln in lines if "/pf/pf_analysis.py final" in ln and not ln.strip().startswith("#")]
        ftypes = sorted(re.search(r"--ftype (\w+)", ln).group(1) for ln in fl if re.search(r"--ftype (\w+)", ln))
        if ftypes != ["global", "mean"]:
            msgs.append(f"final invocations {ftypes}")
        for ln in fl:
            for req_ in ("--trval-feats $PA/feats", "--dev-report $DEV_REPORT", "--ref-meta /pf/smoke-report.json", "--suite libero_goal", "--raw /work/data/libero_goal"):
                if req_ not in ln:
                    msgs.append(f"final line lacks {req_}")
            for cnd in CONDS:
                nm = "clean_rerender" if cnd == "clean" else cnd
                if f"--cond {nm}=$PB/feats/{cnd}" not in ln:
                    msgs.append(f"final line lacks --cond {nm}=$PB/feats/{cnd}")
            if "--stored-feats" in ln:
                msgs.append("final with --stored-feats (stored features are not read in the final run)")
        if "DEV_REPORT=$PA/dev/20261109-144404/dev-report.json" not in script:
            msgs.append("DEV_REPORT is not the phase-A passing report")
        if "-eq 250" not in script or "partial" not in script:
            msgs.append("250-file / .partial check missing")
        if "PFB_SUMMARY_GLOBAL" not in script or "PFB_SUMMARY_MEAN" not in script:
            msgs.append("summary markers missing")
    # --- CLI flags cross-checked against the sources' add_argument strings
    for ln in lines:
        if ln.strip().startswith("#"):
            continue
        for tool in FLAGS:
            if "/" + tool in ln or " " + tool in ln:
                used = set(re.findall(r"(?<![\w$])(--[a-z0-9-]+)", ln.split(tool, 1)[1]))
                unknown = sorted(used - FLAGS[tool] - {"--help"})
                if unknown:
                    msgs.append(f"{tool}: flags not in its add_argument calls: {unknown} in: {ln.strip()[:80]}")
    print(f"{name}: image digest ok={c['image'] == DIGEST}; IMAGE_DIGEST={'set' if envd.get('IMAGE_DIGEST') else 'not set'}; gpu={ngpu}; "
          f"cpu {req['cpu']}/{lim['cpu']}, mem {req['memory']}/{lim['memory']}, eph {req['ephemeral-storage']}/{lim['ephemeral-storage']}; "
          f"bash -n rc={r.returncode}; " + ("OK" if not msgs else "PROBLEMS: " + "; ".join(msgs)))
    bad += bool(msgs)
print(f"Jobs: {len(names)}; GPUs if all ran at once: {total_gpu}")
print("TOTAL problems:", bad)
sys.exit(1 if bad else 0)
