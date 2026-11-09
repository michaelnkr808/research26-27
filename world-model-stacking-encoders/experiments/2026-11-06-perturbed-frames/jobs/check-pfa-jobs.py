#!/usr/bin/env python3
"""pf-jobs-a done-check helper: parse every job-pfa-*.yaml, extract the container script, run bash -n, check the resource and policy rules.
Usage: check.py <jobs dir> <scratch dir>"""
import glob
import re
import subprocess
import sys
from pathlib import Path

import yaml

jobs = Path(sys.argv[1])
scratch = Path(sys.argv[2])
scratch.mkdir(parents=True, exist_ok=True)
DIGEST = "pytorch/pytorch@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2"


def num(v):
    v = str(v)
    m = re.fullmatch(r"([0-9.]+)(m|Ki|Mi|Gi)?", v)
    x, u = float(m.group(1)), m.group(2)
    return x / 1000 if u == "m" else x * {"Ki": 1 / 1024, "Mi": 1, "Gi": 1024}[u] if u else x


bad = 0
for f in sorted(glob.glob(str(jobs / "job-pfa-*.yaml"))):
    name = Path(f).name
    docs = list(yaml.safe_load_all(open(f)))
    assert len(docs) == 1, name
    d = docs[0]
    msgs = []
    spec = d["spec"]
    pod = spec["template"]["spec"]
    c = pod["containers"][0]
    if spec.get("backoffLimit") != 0:
        msgs.append("backoffLimit != 0")
    if pod.get("restartPolicy") != "Never":
        msgs.append("restartPolicy")
    if "tolerations" in pod:
        msgs.append("tolerations present")
    if c["image"] != DIGEST:
        msgs.append("image not the pinned digest")
    envd = {e["name"]: e.get("value") for e in c["env"]}
    render = "render" in name or "pilot" in name
    if render and envd.get("IMAGE_DIGEST") != c["image"]:
        msgs.append("IMAGE_DIGEST env differs from the image")
    req, lim = c["resources"]["requests"], c["resources"]["limits"]
    cpu_only = "dev" in name
    ngpu = int(req.get("nvidia.com/gpu", 0))
    if ngpu != (0 if cpu_only else 1) or int(lim.get("nvidia.com/gpu", 0)) != ngpu:
        msgs.append(f"gpu request {ngpu}")
    for k in ("cpu", "memory", "ephemeral-storage"):
        r, l = num(req[k]), num(lim[k])
        if not (r <= l <= 1.2 * r + 1e-9):
            msgs.append(f"{k}: request {req[k]} limit {lim[k]} ratio {l / r:.3f}")
    if not any(v["name"] == "dshm" for v in pod["volumes"]) and not cpu_only:
        msgs.append("no /dev/shm emptyDir")
    aff = str(pod["affinity"])
    if "us-west" not in aff:
        msgs.append("no us-west")
    if not cpu_only and "ry-gpu-08" not in aff:
        msgs.append("excluded hosts missing")
    if cpu_only and "pci-10de" not in aff:
        msgs.append("no GPU-node anti-affinity")
    if "render" in name or "pilot" in name:
        if "['NVIDIA-L4']" not in aff:
            msgs.append("render not L4 only")
    if spec.get("ttlSecondsAfterFinished") is None or not spec.get("activeDeadlineSeconds"):
        msgs.append("ttl or deadline missing")
    script = c["args"][0]
    sp = scratch / (Path(f).stem + ".sh")
    sp.write_text(script)
    r = subprocess.run(["bash", "-n", str(sp)], capture_output=True, text=True)
    if r.returncode != 0:
        msgs.append("bash -n: " + r.stderr.strip()[:300])
    if re.search(r"\bsleep\b", script):
        msgs.append("sleep in script")
    if re.search(r"\bwhile\b|\buntil\b", script):
        msgs.append("loop keyword while/until in script")
    print(f"{name}: image digest ok={c['image'] == DIGEST}; IMAGE_DIGEST={'set' if envd.get('IMAGE_DIGEST') else 'not set'}; gpu={ngpu}; "
          f"cpu {req['cpu']}/{lim['cpu']}, mem {req['memory']}/{lim['memory']}, eph {req['ephemeral-storage']}/{lim['ephemeral-storage']}; "
          f"bash -n rc={r.returncode}; " + ("OK" if not msgs else "PROBLEMS: " + "; ".join(msgs)))
    bad += bool(msgs)
print("TOTAL problems:", bad)
sys.exit(1 if bad else 0)
