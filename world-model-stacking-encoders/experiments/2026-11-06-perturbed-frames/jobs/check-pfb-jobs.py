#!/usr/bin/env python3
"""pfb-jobs done-check helper: parse every job-pfb-*.yaml (nine expected, plus the optional RTX A6000 twin of the render Job, job-pfb-render-a6000.yaml, plus the recovery Job job-pfb-recover.yaml), extract the container script, run bash -n, check the resource, policy and phase-B rules.
Usage: check-pfb-jobs.py [--rendered [--any-values]] <jobs dir> <scratch dir> [<src dir>]    (src dir default: ../../../src of this file; used to cross-check the CLI flags the scripts pass)
With --rendered (the recovery copy must then carry the A6000 render copy's commit and NOISE_CAP, unless --any-values) a manifest may be a submit-time copy: SRC_GIT_COMMIT must then be 40 hex characters and NOISE_CAP (render-type Jobs) digits, with no FILL_AT_SUBMIT left in a value; a manifest that still has the placeholders is checked as before.
Started from check-pfa-jobs.py (same checks), plus the phase-B specifics: pinned digest in every Job and IMAGE_DIGEST equal to it in the render Job; per-condition staging paths;
--consistency-ref on every test render and --expect-draws on every perturbed one; --config /pf/phase-a.yaml everywhere; revision asserts; no sleep/while/until; CPU Jobs without GPU."""
import glob
import re
import subprocess
import sys
from pathlib import Path

import yaml

RENDERED = "--rendered" in sys.argv
ANY_VALUES = "--any-values" in sys.argv    # accept other commit / noise cap values in a rendered recovery copy (tests with dummy values)
sys.argv = [a for a in sys.argv if a not in ("--rendered", "--any-values")]
RENDERED_COMMIT = "a3777d7db734bf1a7275df17b0e5f4ae57e90c84"    # jobs/rendered/job-pfb-render-a6000.yaml (the run whose clean DINOv3 features exist)
RENDERED_NOISE_CAP = "10800"
jobs = Path(sys.argv[1])
scratch = Path(sys.argv[2])
scratch.mkdir(parents=True, exist_ok=True)
src_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).resolve().parents[3] / "src"
DIGEST = "pytorch/pytorch@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2"
CONFIGMAP = "stacking-encoders-pf-main-src-b"
CONDS = ["clean", "camera", "light", "background", "noise"]
EXPECTED = (["job-pfb-draws.yaml", "job-pfb-render.yaml", "job-pfb-ext-vggt.yaml"] + [f"job-pfb-ext-sam3-{c}.yaml" for c in CONDS] + ["job-pfb-final.yaml"])
CPU_ONLY = {"job-pfb-draws.yaml", "job-pfb-final.yaml"}
TWIN = "job-pfb-render-a6000.yaml"    # 2026-11-10: RTX A6000 twin of the render Job, raced against the original; never counted as a second GPU (see PHASE-B.md note 2026-11-10)
TWIN_RESOURCE = "nvidia.com/rtxa6000"
ORIGINAL = "job-pfb-render.yaml"
T3090_RESOURCE = "nvidia.com/GA102_GEFORCE_RTX_3090"
T3090 = {f"job-pfb-ext-sam3-{c}-3090.yaml": f"job-pfb-ext-sam3-{c}.yaml" for c in ("background", "noise")}    # 2026-11-11: RTX 3090 twins (typed resource) of two SAM 3 Jobs, raced against the originals (PHASE-B.md section 12)
RECOVER = "job-pfb-recover.yaml"      # 2026-11-10: one combined GPU Job that recovers what the failed A6000 render Job did not finish (PHASE-B.md section 11); replaces the render Job, never runs beside it
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
extra = sorted(Path(f).name for f in glob.glob(str(jobs / "job-pfb-*.yaml")) if Path(f).name not in EXPECTED and Path(f).name not in (TWIN, RECOVER) and Path(f).name not in T3090)
if missing or extra:
    print("PROBLEMS: expected manifests missing", missing, "unexpected", extra)
    bad += 1
names = set()
total_gpu = 0
docs_by_name = {}
for f in sorted(glob.glob(str(jobs / "job-pfb-*.yaml"))):
    name = Path(f).name
    docs = list(yaml.safe_load_all(open(f)))
    msgs = []
    if len(docs) != 1:
        print(f"{name}: {len(docs)} documents")
        bad += 1
        continue
    d = docs[0]
    docs_by_name[name] = d
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
    is_twin = name == TWIN
    is_recover = name == RECOVER
    is_t3 = name in T3090
    is_render = name in (ORIGINAL, TWIN, RECOVER)    # the twin runs the same script, so every render check applies to it; the recovery Job shares the manifest checks and has its own script checks below
    if is_render and envd.get("IMAGE_DIGEST") != c["image"]:
        msgs.append("IMAGE_DIGEST env differs from the image")
    if is_render and envd.get("IMAGE_DIGEST") != DIGEST:
        msgs.append("IMAGE_DIGEST is not the pinned digest")
    filled = RENDERED and envd.get("SRC_GIT_COMMIT") != "FILL_AT_SUBMIT"    # a submit-time copy (comments may still say FILL_AT_SUBMIT; values must not)
    if filled:
        if not re.fullmatch(r"[0-9a-f]{40}", str(envd.get("SRC_GIT_COMMIT"))):
            msgs.append(f"rendered: SRC_GIT_COMMIT env {envd.get('SRC_GIT_COMMIT')!r} is not 40 hex characters")
        if is_recover and not ANY_VALUES:
            if envd.get("SRC_GIT_COMMIT") != RENDERED_COMMIT:
                msgs.append(f"rendered recover: SRC_GIT_COMMIT is not the A6000 render copy's {RENDERED_COMMIT} (--any-values to allow)")
            if not re.search(r"^ *NOISE_CAP=" + RENDERED_NOISE_CAP + "$", c["args"][0], re.M):
                msgs.append(f"rendered recover: NOISE_CAP is not the A6000 render copy's {RENDERED_NOISE_CAP} (--any-values to allow)")
    elif envd.get("SRC_GIT_COMMIT") != "FILL_AT_SUBMIT":
        msgs.append("SRC_GIT_COMMIT env is not FILL_AT_SUBMIT")
    if envd.get("PYTHONPATH") != "/pf:/src":
        msgs.append(f"PYTHONPATH env {envd.get('PYTHONPATH')!r}")
    req, lim = c["resources"]["requests"], c["resources"]["limits"]
    cpu_only = name in CPU_ONLY
    if is_t3:
        ngpu = int(req.get(T3090_RESOURCE, 0))
        if ngpu != 1 or int(lim.get(T3090_RESOURCE, 0)) != 1 or "nvidia.com/gpu" in req or "nvidia.com/gpu" in lim or TWIN_RESOURCE in req or TWIN_RESOURCE in lim:
            msgs.append(f"3090 twin must request exactly {T3090_RESOURCE}: 1 and no other GPU resource")
    elif is_twin:
        # the twin asks for exactly one RTX A6000 under its own resource name and no generic GPU
        ngpu = int(req.get(TWIN_RESOURCE, 0))
        if ngpu != 1 or int(lim.get(TWIN_RESOURCE, 0)) != 1 or "nvidia.com/gpu" in req or "nvidia.com/gpu" in lim:
            msgs.append(f"twin must request exactly {TWIN_RESOURCE}: 1 and no nvidia.com/gpu")
    else:
        ngpu = int(req.get("nvidia.com/gpu", 0))
        if ngpu != (0 if cpu_only else 1) or int(lim.get("nvidia.com/gpu", 0)) != ngpu or TWIN_RESOURCE in req or TWIN_RESOURCE in lim or T3090_RESOURCE in req or T3090_RESOURCE in lim:
            msgs.append(f"gpu request {ngpu}")
    if not is_twin and not is_recover and not is_t3:    # the twin and the original are raced and never both run, so only the original counts toward the peak; the recovery Job replaces the render Job
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
    req_terms = pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
    req_exprs = [me for t in req_terms for me in t.get("matchExpressions", [])]
    if is_render and len(req_terms) != 1:
        msgs.append(f"render manifest must have exactly one required nodeSelectorTerm (terms are ORed), found {len(req_terms)}")
    has_req_region = any(me.get("key") == "topology.kubernetes.io/region" for me in req_exprs)
    if is_t3:
        # 3090 twin: no gpu.product, no driver rule, no required region, excluded hosts kept (checked below), soft us-west preference weight 100
        if any(me.get("key") == "nvidia.com/gpu.product" for me in req_exprs):
            msgs.append("3090 twin must not require nvidia.com/gpu.product")
        if any(me.get("key") == "nvidia.com/cuda.driver.major" for me in req_exprs):
            msgs.append("3090 twin must not carry the driver label rule (typed-resource nodes have no such label; the script checks the driver instead)")
        if has_req_region:
            msgs.append("3090 twin must not require a region (soft preference only)")
        prefs = pod["affinity"]["nodeAffinity"].get("preferredDuringSchedulingIgnoredDuringExecution", [])
        if not any(pr.get("weight") == 100 and any(me.get("key") == "topology.kubernetes.io/region" and me.get("operator") == "In" and me.get("values") == ["us-west"]
                                                    for me in pr["preference"].get("matchExpressions", [])) for pr in prefs):
            msgs.append("3090 twin lacks the preferred us-west term (weight 100)")
        if "priorityClassName" in pod:
            msgs.append("priorityClassName present")
    elif is_twin:
        # twin: no gpu.product, no required region, driver rule and excluded hosts kept, soft us-west preference weight 100
        if any(me.get("key") == "nvidia.com/gpu.product" for me in req_exprs):
            msgs.append("twin must not require nvidia.com/gpu.product")
        if has_req_region:
            msgs.append("twin must not require a region (soft preference only)")
        if not any(me.get("key") == "nvidia.com/cuda.driver.major" and me.get("operator") == "Gt" and me.get("values") == ["569"] for me in req_exprs):
            msgs.append("twin lost the driver rule")
        prefs = pod["affinity"]["nodeAffinity"].get("preferredDuringSchedulingIgnoredDuringExecution", [])
        if not any(pr.get("weight") == 100 and any(me.get("key") == "topology.kubernetes.io/region" and me.get("operator") == "In" and me.get("values") == ["us-west"]
                                                    for me in pr["preference"].get("matchExpressions", [])) for pr in prefs):
            msgs.append("twin lacks the preferred us-west term (weight 100)")
    elif not cpu_only and not any(me.get("key") == "topology.kubernetes.io/region" and me.get("operator") == "In" and me.get("values") == ["us-west"] for me in req_exprs):
        msgs.append("us-west is not a REQUIRED rule")
    if not cpu_only and "ry-gpu-08" not in aff:
        msgs.append("excluded hosts missing")
    if cpu_only and "pci-10de" not in aff:
        msgs.append("no GPU-node anti-affinity")
    if cpu_only and "nvidia.com/gpu.product" in aff:
        msgs.append("CPU job selects a GPU product")
    if name in (ORIGINAL, RECOVER):
        render_gpus = set()
        for term in pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]:
            for me in term.get("matchExpressions", []):
                if me.get("key") == "nvidia.com/gpu.product" and me.get("operator") == "In":
                    render_gpus |= set(me["values"])
        if render_gpus != {"NVIDIA-L4", "NVIDIA-A10", "NVIDIA-RTX-A5000", "NVIDIA-GeForce-RTX-3090", "NVIDIA-GeForce-RTX-4090"}:
            msgs.append("render GPU set is not the five 24 GB cards")
    if spec.get("ttlSecondsAfterFinished") is None or not spec.get("activeDeadlineSeconds"):
        msgs.append("ttl or deadline missing")
    script = c["args"][0]
    if filled and re.search(r"=FILL_AT_SUBMIT\b|value: FILL_AT_SUBMIT", script + str(c["env"])):
        msgs.append("rendered: a FILL_AT_SUBMIT placeholder is still used as a value")
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
        if seen != ({"noise": 1} if is_recover else {c_: 1 for c_ in CONDS}):
            msgs.append(f"render conditions {seen}" + (" (the recovery Job may render noise only)" if is_recover else ""))
        if "PA=$R/pf-main/phase-a" not in script or "R=/mnt/proxy" not in script:
            msgs.append("PA is not /mnt/proxy/pf-main/phase-a")
        if not re.search(r"^ *NOISE_CAP=" + (r"[0-9]+" if filled else "FILL_AT_SUBMIT") + "$", script, re.M):
            msgs.append("NOISE_CAP variable missing or not " + ("a number (rendered copy)" if filled else "the FILL_AT_SUBMIT placeholder"))
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
    if is_recover:
        # the recovery script is compared with the render Job's script as text: the extraction block, the meta and copy stages and the noise render line must be the render Job's, apart from the documented additions
        o_path = jobs / ORIGINAL
        if not o_path.is_file():
            msgs.append("recover: job-pfb-render.yaml missing, cannot compare the script")
        else:
            ol = [ln.strip() for ln in yaml.safe_load(open(o_path))["spec"]["template"]["spec"]["containers"][0]["args"][0].split("\n")]
            rl = [ln.strip() for ln in script.split("\n")]

            def arms(lns):
                i_ = next((k for k, ln in enumerate(lns) if ln.startswith("case $_e in")), None)
                j_ = next((k for k in range(i_ + 1, len(lns)) if lns[k] == "esac"), None) if i_ is not None else None
                return None if i_ is None or j_ is None else lns[i_:j_ + 1]
            if arms(ol) is None or arms(ol) != arms(rl):
                msgs.append("recover: the encoder arms (python, flags, batch size, caps) differ from the render Job's")
            for key in ("stage meta-$_e-$_c", "stage copy-$_e-$_c", "if [ $LAST -eq 0 ]; then rm -rf /work/feats/$_c/$_e; fi"):
                if [ln for ln in ol if ln.startswith(key)] != [ln for ln in rl if ln.startswith(key)] or not [ln for ln in rl if ln.startswith(key)]:
                    msgs.append(f"recover: the line starting {key!r} differs from the render Job's or is missing")
            if [ln for ln in ol if ln.startswith("stage render-noise")] != [ln for ln in rl if ln.startswith("stage render-noise")]:
                msgs.append("recover: the noise render line differs from the render Job's")
            o_ext = [ln for ln in ol if ln.startswith("stage extract-$_e-$_c cap $_t ")]
            r_ext = [ln for ln in rl if ln.startswith("cap $_t $_py /src/extract_features.py")]
            if len(o_ext) != 1 or len(r_ext) != 1:
                msgs.append("recover: cannot find the extraction command in both scripts")
            else:
                r_norm = re.sub(r" 2>&1 \| tee \S+$", "", r_ext[0]).replace(" --device cuda", "")
                if "stage extract-$_e-$_c " + r_norm != o_ext[0]:
                    msgs.append("recover: the extraction command differs from the render Job's beyond --device cuda and the log tee")
        # published renders are only read: any line that writes toward render/ or a published folder must be a line of the render Job (publish_render) or the one staging copy
        write_verb = re.compile(r"\b(cp|mv|mkdir|touch|rm|ln|tee|dd|truncate)\b|(?<![0-9&])>\s*\S*\$(PB/render|_d\b)")
        for ln in rl:
            if ln.startswith("#") or not ("$PB/render" in ln or "$_d" in ln):
                continue
            if write_verb.search(ln) and ln not in ol and ln != "cp $_d/libero_goal/*.hdf5 $_l/":
                msgs.append("recover: a line writes toward render/ or a published folder and is not a render Job line: " + ln[:100])
        for ln in rl:
            if re.search(r"\btouch\b", ln) and not ln.startswith("#") and ln not in ol:
                msgs.append("recover: a touch that the render Job does not have: " + ln[:100])
        if not re.search(r'^ *PUBC="clean camera light background"$', script, re.M):
            msgs.append("recover: PUBC is not exactly the four published conditions")
        pre = script.split("for c in $CONDS; do", 1)[-1].split("\ndone\n", 1)[0]    # the pre-flight loop only
        for x in ("exit 20", "exit 14", "exit 22"):
            if x not in pre:
                msgs.append(f"recover: {x} refusal missing from the pre-flight loop")
        pr_lines = [ln.strip() for ln in lines if "publish_render" in ln and not ln.strip().startswith("#")]
        if len(pr_lines) != 3 or not any(ln.startswith("publish_render()") for ln in pr_lines) or not any(ln.startswith("export -f publish_render") for ln in pr_lines) \
                or not any(ln.startswith("stage publish-$_c cap 1800 bash -c 'publish_render") for ln in pr_lines):
            msgs.append(f"recover: publish_render is used outside its definition, export and the stage in finish_render: {pr_lines}")
        if 'REFUSE: $PB/render/$_c or $_c.part exists' not in script:
            msgs.append("recover: publish_render does not refuse an existing target")
        if "sha256sum -c $_d/render-sha256.txt" not in script or "RENDER_COMPLETE" not in script:
            msgs.append("recover: staging is not checked against render-sha256.txt / RENDER_COMPLETE")
        if not re.search(r"elif \[ -f \$PB/render/\$c/RENDER_COMPLETE \] && \[ -f \$PB/render/\$c/render-sha256.txt \] && \[ -f \$PB/render/\$c/render-report.json \]", script):
            msgs.append("recover: the 'noise already published' pre-flight does not require RENDER_COMPLETE, render-sha256.txt and render-report.json")
        for ln in lines:
            if not ln.strip().startswith("#") and re.search(r"\brm\b", ln) and ("$PB" in ln or "/mnt/proxy" in ln or "$R" in ln):
                msgs.append("recover: rm on the volume: " + ln.strip()[:90])
        if "feats_done $PB/feats/$_c/$_e/libero_goal" not in script or "meta_check.py $_e $_c $PB/feats/$_c" not in script:
            msgs.append("recover: skipped complete features are not meta-checked")
        # order (the user's decision 2026-11-10, after the idle-GPU analysis): extractions of the published conditions first, then the noise render, then the noise extractions; exit 5 / 3 of the noise render end the Job
        i_pub = script.find('for e in dinov3 siglip2 c-radiov4; do for c in $CONDS; do if [ "${MODE[$c]}" = pub ]; then extract_one $e $c; fi; done; done')
        i_noise = script.find("stage render-noise")
        i_next = script.find('if [ "${MODE[noise]:-}" = new ]; then for e in dinov3 siglip2 c-radiov4; do extract_one $e noise; done; fi')
        if min(i_pub, i_noise, i_next) < 0 or not (i_pub < i_noise < i_next):
            msgs.append("recover: the order is not published extractions, then the noise render, then the noise extractions")
        if "[ $_nrc -eq 5 ] || [ $_nrc -eq 3 ]" not in script or 'abort_job $_nrc' not in script:
            msgs.append("recover: exit 5 / 3 of the noise render do not end the Job")
        ab = script.split("abort_job() {", 1)[-1].split("\n}", 1)[0]
        if "BEGIN PFB_TGZ_B64" not in ab or "base64 -w 76" not in ab or "exit $1" not in ab:
            msgs.append("recover: abort_job does not print the report tarball")
        # dead-GPU fail-fast: checked before every extraction stage, in the stage's own python; device=cpu ends the Job; extraction forced to cuda
        gc = script.split("gpu_check() {", 1)[-1][:700] if "gpu_check() {" in script else ""
        if "nvidia-smi -L" not in gc or "torch.cuda.is_available()" not in gc or "torch.ones(64, device=\"cuda\")" not in gc or "$_py -c" not in gc or "abort_job 30" not in gc:
            msgs.append("recover: gpu_check is missing or incomplete (nvidia-smi, torch.cuda.is_available, a CUDA op in $_py, abort_job 30)")
        i_g = script.find("gpu_check extract-$_e-$_c\n")
        i_s = script.find("stage extract-$_e-$_c run_extract")
        if i_g < 0 or i_s < 0 or i_g > i_s:
            msgs.append("recover: gpu_check is not called right before the extraction stage")
        if "grep -q 'device=cpu'" not in script or "abort_job 31" not in script:
            msgs.append("recover: a device=cpu log does not end the Job (exit 31)")
        ext_cmd = [ln for ln in rl if ln.startswith("cap $_t $_py /src/extract_features.py")]    # the actual command line, not a comment
        if len(ext_cmd) != 1 or not re.search(r" --device cuda( |$)", ext_cmd[0]):
            msgs.append("recover: the extraction command line itself does not carry --device cuda")
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
            cond = name[len("job-pfb-ext-sam3-"):-len(".yaml")].removesuffix("-3090")
            if conds != [cond]:
                msgs.append(f"{name}: CONDS {conds}, expected [{cond}]")
            if enc != "sam3" or "3c879f3" not in script or "facebook/sam3" not in script:
                msgs.append("sam3: encoder or revision assert")
            if "transformers==5.18.0" not in script:
                msgs.append("sam3: pins")
        if is_t3:
            # the driver fail-fast comes before the free-space check and every install; CUDA check and --device cuda; a CPU-fallback guard on the log
            i_drv = script.find("exit 13")
            if "nvidia-smi --query-gpu=driver_version" not in script or i_drv < 0 or not re.search(r"-le 569 \]; then .*exit 13; fi", script):
                msgs.append("3090 twin: driver fail-fast (exit 13 when the driver major is 569 or lower) missing")
            elif not (i_drv < script.find("df -BG") and i_drv < script.find("pip install")):
                msgs.append("3090 twin: the driver fail-fast does not come before the free-space check and the installs")
            if "torch.cuda.is_available()" not in script or "exit 30" not in script:
                msgs.append("3090 twin: CUDA check (exit 30) missing")
            if "--device cuda" not in script:
                msgs.append("3090 twin: extractor is not called with --device cuda")
            if not re.search(r"grep -q 'device=cpu' \$LOGF", script) or "WORST=31" not in script:
                msgs.append("3090 twin: device=cpu guard (worst code 31) missing")
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
if TWIN in docs_by_name and ORIGINAL in docs_by_name:
    # the twin must be the original with only the allowed differences: same script, same env, same volumes, same CPU, memory and ephemeral storage
    a, b = docs_by_name[ORIGINAL], docs_by_name[TWIN]
    ca, cb = a["spec"]["template"]["spec"]["containers"][0], b["spec"]["template"]["spec"]["containers"][0]
    tmsgs = []
    def norm_pod(d_):
        """The whole pod spec without the affinity and without the GPU resource key, for the twin comparison."""
        import copy
        ps = copy.deepcopy(d_["spec"]["template"]["spec"])
        ps.pop("affinity", None)
        for sect in ("requests", "limits"):
            for k in ("nvidia.com/gpu", TWIN_RESOURCE):
                ps["containers"][0]["resources"][sect].pop(k, None)
        return ps
    if norm_pod(a) != norm_pod(b):
        tmsgs.append("pod spec differs beyond affinity and the GPU resource key (for example a pod-level nodeSelector or priorityClassName)")
    if a["spec"]["template"].get("metadata") != b["spec"]["template"].get("metadata"):
        tmsgs.append("pod template metadata differs")

    def hosts(d_):
        t = d_["spec"]["template"]["spec"]["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
        return sorted(v for term in t for me in term.get("matchExpressions", []) if me.get("key") == "kubernetes.io/hostname" and me.get("operator") == "NotIn" for v in me["values"])
    if hosts(a) != hosts(b):
        tmsgs.append(f"excluded hosts differ: original {hosts(a)}, twin {hosts(b)}")
    if ca["args"] != cb["args"]:
        tmsgs.append("script differs from the original")
    if ca["env"] != cb["env"] or ca["image"] != cb["image"] or ca["command"] != cb["command"] or ca["volumeMounts"] != cb["volumeMounts"]:
        tmsgs.append("env, image, command or volumeMounts differ")
    if a["spec"]["template"]["spec"]["volumes"] != b["spec"]["template"]["spec"]["volumes"]:
        tmsgs.append("volumes differ")
    for sect in ("requests", "limits"):
        ra = {k: v for k, v in ca["resources"][sect].items() if k != "nvidia.com/gpu"}
        rb = {k: v for k, v in cb["resources"][sect].items() if k != TWIN_RESOURCE}
        if ra != rb:
            tmsgs.append(f"{sect} differ beyond the GPU resource name")
    if {k: v for k, v in a["spec"].items() if k != "template"} != {k: v for k, v in b["spec"].items() if k != "template"}:
        tmsgs.append("Job spec (deadline, ttl, backoff) differs")
    print("twin vs original:", "identical apart from name, GPU resource name and affinity" if not tmsgs else "PROBLEMS: " + "; ".join(tmsgs))
    bad += bool(tmsgs)
for t_, o_ in T3090.items():
    if t_ in docs_by_name and o_ in docs_by_name:
        # the 3090 twin is the original with only the allowed differences: name, GPU resource key, affinity, and the script lines that fail fast
        a, b = docs_by_name[o_], docs_by_name[t_]
        ca, cb = a["spec"]["template"]["spec"]["containers"][0], b["spec"]["template"]["spec"]["containers"][0]
        tmsgs = []
        import copy
        def norm3(d_):
            ps = copy.deepcopy(d_["spec"]["template"]["spec"])
            ps.pop("affinity", None)
            ps["containers"][0].pop("args")
            for sect in ("requests", "limits"):
                for k in ("nvidia.com/gpu", T3090_RESOURCE):
                    ps["containers"][0]["resources"][sect].pop(k, None)
            return ps
        if norm3(a) != norm3(b):
            tmsgs.append("pod spec differs beyond affinity, script and the GPU resource key")
        if {k: v for k, v in a["spec"].items() if k != "template"} != {k: v for k, v in b["spec"].items() if k != "template"}:
            tmsgs.append("Job spec (deadline, ttl, backoff) differs")
        if a["spec"]["template"].get("metadata") != b["spec"]["template"].get("metadata"):
            tmsgs.append("pod template metadata differs")
        def hosts3(d_):
            t = d_["spec"]["template"]["spec"]["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
            return sorted(v for term in t for me in term.get("matchExpressions", []) if me.get("key") == "kubernetes.io/hostname" and me.get("operator") == "NotIn" for v in me["values"])
        if not hosts3(b) or hosts3(a) != hosts3(b):
            tmsgs.append("excluded hosts differ or are missing")
        for sect in ("requests", "limits"):
            ra = {k: v for k, v in ca["resources"][sect].items() if k != "nvidia.com/gpu"}
            rb = {k: v for k, v in cb["resources"][sect].items() if k != T3090_RESOURCE}
            if ra != rb:
                tmsgs.append(f"{sect} differ beyond the GPU resource name")
        # script: every line of the original is in the twin except three edited ones; the twin adds three edited lines and six new ones
        la = ca["args"][0].split("\n")
        lb = cb["args"][0].split("\n")
        missing_lines = [ln for ln in la if ln not in lb]
        extra_lines = [ln for ln in lb if ln not in la]
        if len(missing_lines) != 3 or len(extra_lines) != 9:
            tmsgs.append(f"script: {len(missing_lines)} original lines changed (expected 3) and {len(extra_lines)} lines new (expected 9)")
        print(f"{t_} vs {o_}:", "identical apart from name, GPU resource name, affinity and the fail-fast script lines" if not tmsgs else "PROBLEMS: " + "; ".join(tmsgs))
        bad += bool(tmsgs)
if RECOVER in docs_by_name and ORIGINAL in docs_by_name:
    # the recovery Job is the render Job with another name and another script: everything else (node rule, resources, env, volumes, deadline, ttl) must be identical
    a, b = docs_by_name[ORIGINAL], docs_by_name[RECOVER]
    rmsgs = []
    import copy
    def norm(d_):
        d2 = copy.deepcopy(d_)
        d2["metadata"].pop("name")
        d2["spec"]["template"]["spec"]["containers"][0].pop("args")
        for e_ in d2["spec"]["template"]["spec"]["containers"][0]["env"]:
            if e_["name"] == "SRC_GIT_COMMIT":
                e_["value"] = None    # a submit-time copy carries the filled commit (checked above)
        return d2
    if norm(a) != norm(b):
        rmsgs.append("manifest differs from the render Job beyond name and script (node rule, resources, env, volumes, deadline or ttl)")
    print("recover vs original:", "identical apart from name and script" if not rmsgs else "PROBLEMS: " + "; ".join(rmsgs))
    bad += bool(rmsgs)
print(f"Jobs: {len(names)}; GPUs if all ran at once: {total_gpu}" + (" (the A6000 twin is not counted: it races the original and the loser is deleted, so at most one of the pair runs)" if TWIN in docs_by_name else "") + (" (the two RTX 3090 twins are not counted: each races its original and the loser is deleted)" if any(t_ in docs_by_name for t_ in T3090) else "") + (" (the recovery Job is not counted: it replaces the render Job)" if RECOVER in docs_by_name else ""))
print("TOTAL problems:", bad)
sys.exit(1 if bad else 0)
