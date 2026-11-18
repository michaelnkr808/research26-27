#!/usr/bin/env python3
"""Overnight watcher for the suspended Job stacking-encoders-pairings-siglip2-reextract (2026-11-18).
Waits until NRP's Ceph 'full' alarms (OSD_FULL, POOL_FULL, cluster 'rook') have been off for CLEAR_H hours,
resumes the Job (spec.suspend=false), then watches it: saves pod logs, a 15-minute GPU check, and suspends again
on a new pod failure, on a returning alarm, or on an idle GPU. Never deletes anything. Stdlib only."""
import datetime, json, os, re, subprocess, sys, time

NS, JOB = "aria-world", "stacking-encoders-pairings-siglip2-reextract"
PROM = "https://prometheus.nrp-nautilus.io/api/v1/query"
D, RES = sys.argv[1], sys.argv[2]
DRY = "--dry" in sys.argv
ONCE = "--once" in sys.argv
CLEAR_H = 2          # hours the alarm must have been off before a resume
MAX_RESUMES = 4
HARD_END = time.time() + 20 * 3600
ALARM = 'ceph_health_detail{namespace="rook",name=~"OSD_FULL|POOL_FULL"}'
st = {"mode": "WAIT_STORAGE", "resumes": 0, "kc_dead_until": 0, "failed_base": None, "pods": {}}

def now(): return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def _w(name, msg):
    with open(os.path.join(D, name), "a") as f: f.write(f"{now()} {msg}\n")
def status(msg): _w("status.log", msg)
def event(msg): _w("events.log", msg); _w("status.log", "EVENT " + msg)

def kc(*args, timeout=90):
    """kubectl with a hard timeout. An expired OIDC login makes kubectl hang: back off for an hour."""
    if time.time() < st["kc_dead_until"]: return None
    try:
        r = subprocess.run(["kubectl", *args], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        st["kc_dead_until"] = time.time() + 3600
        event("KUBECTL-HUNG (login probably expired); no kubectl for 60 min. Fix: ! kubectl get pods -n aria-world")
        return None
    if r.returncode != 0:
        status(f"kubectl {' '.join(args[:3])} rc={r.returncode}: {r.stderr.strip()[:300]}")
        return None
    return r.stdout

def prom(query):
    try:
        r = subprocess.run(["curl", "-s", "-m", "40", "--data-urlencode", "query=" + query, PROM],
                           capture_output=True, text=True, timeout=60)
        j = json.loads(r.stdout)
        return j["data"]["result"] if j.get("status") == "success" else None
    except Exception as e:
        status(f"prometheus failed: {e!r}"[:300]); return None

def alarm(window):
    """(seen, detail): seen is True if either alarm was on in the window, False if both were off with enough
    samples, None if Prometheus gave no usable answer (treated as 'not clear')."""
    mx, ct = prom(f"max_over_time({ALARM}[{window}])"), prom(f"count_over_time({ALARM}[{window}])")
    if not mx or not ct or len(mx) < 2 or len(ct) < 2: return None, "no data"
    v = {x["metric"]["name"]: float(x["value"][1]) for x in mx}
    c = {x["metric"]["name"]: float(x["value"][1]) for x in ct}
    need = {"10m": 5, f"{CLEAR_H}h": 60 * CLEAR_H // 2}.get(window, 5)   # at least one sample per 2 minutes
    if set(v) != {"OSD_FULL", "POOL_FULL"} or min(c.values()) < need: return None, f"thin data max={v} count={c}"
    return any(x > 0 for x in v.values()), f"max={v} count={c}"

def job():
    o = kc("get", "job", JOB, "-n", NS, "-o", "json")
    return json.loads(o) if o else None

def pods():
    o = kc("get", "pods", "-n", NS, "-l", "job-name=" + JOB, "-o", "json")
    return json.loads(o)["items"] if o else None

def save_log(name, final):
    path = os.path.join(RES, name + ".log")
    if final and os.path.exists(path) and st["pods"].get(name, {}).get("final"): return path
    o = kc("logs", name, "-n", NS, timeout=180)
    if o is None: return None
    with open(path, "w") as f: f.write(o)
    if final: st["pods"].setdefault(name, {})["final"] = True
    return path

def set_suspend(flag, why):
    if DRY: event(f"DRY: would set suspend={flag} ({why})"); return True
    o = kc("patch", "job", JOB, "-n", NS, "--type=merge", "-p", json.dumps({"spec": {"suspend": flag}}))
    event(f"{'SUSPENDED' if flag else 'RESUMED'} ({why})" if o is not None else f"PATCH-FAILED suspend={flag} ({why})")
    return o is not None

def suspend_running(why, ps):
    for p in ps or []:
        if p["status"].get("phase") == "Running": save_log(p["metadata"]["name"], False)
    return set_suspend(True, why)

def gpu_vals(text): return [int(m) for m in re.findall(r"^\[gpu\] [^,]+, (\d+) %", text, re.M)]

def wait_storage():
    seen, detail = alarm(f"{CLEAR_H}h")
    status(f"WAIT_STORAGE alarm_seen_{CLEAR_H}h={seen} {detail} resumes={st['resumes']}")
    if seen is not False: return 600
    j = job()
    if j is None: return 600
    if not j["spec"].get("suspend"):
        event("Job is not suspended any more (someone resumed it); watching it"); st["mode"] = "RUNNING"
        st["failed_base"] = j.get("status", {}).get("failed", 0); return 60
    if st["resumes"] >= MAX_RESUMES:
        event(f"HOLD: {MAX_RESUMES} resumes used; Job stays suspended"); return None
    if set_suspend(False, f"alarms off for {CLEAR_H} h: {detail}"):
        st["resumes"] += 1; st["mode"] = "RUNNING"; st["failed_base"] = j.get("status", {}).get("failed", 0)
        return 120 if not DRY else None
    return 600

def running():
    j = job()
    if j is None:
        r = prom(f'kube_pod_status_phase{{namespace="{NS}",pod=~"{JOB}-.*"}} == 1')
        status("RUNNING (no kubectl) prom phases: " + (", ".join(f"{x['metric']['pod'][-7:]}={x['metric']['phase']}" for x in r) if r is not None else "none"))
        return 300
    ps = pods() or []
    s = j.get("status", {})
    conds = {c["type"]: c for c in s.get("conditions", []) if c.get("status") == "True"}
    line = ", ".join(f"{p['metadata']['name'][-7:]}={p['status'].get('phase')}@{p['spec'].get('nodeName', '-')}" for p in ps)
    status(f"RUNNING active={s.get('active', 0)} succeeded={s.get('succeeded', 0)} failed={s.get('failed', 0)} pods: {line}")
    for p in ps:   # terminal pods: keep their logs
        n, ph = p["metadata"]["name"], p["status"].get("phase")
        if ph in ("Succeeded", "Failed") and not st["pods"].get(n, {}).get("final"):
            path = save_log(n, True)
            tail = open(path).read()[-400:].replace("\n", " | ") if path else "log not saved"
            event(f"POD-{ph.upper()} {n}: {tail}")
    if "Complete" in conds or "Failed" in conds:
        o = kc("get", "job", JOB, "-n", NS, "-o", "yaml")
        if o: open(os.path.join(RES, "job-status.yaml"), "w").write(o)
        c = conds.get("Complete") or conds.get("Failed")
        event(f"JOB-{c['type'].upper()} reason={c.get('reason')} {c.get('message', '')[:200]}. Not deleted (RUN step 7 is the lead's).")
        return None
    if j["spec"].get("suspend"):
        st["mode"] = "WAIT_STORAGE"; return 600
    if s.get("failed", 0) > (st["failed_base"] or 0):
        st["failed_base"] = s.get("failed", 0)
        suspend_running(f"a pod failed after the resume (failed={s.get('failed')})", ps)
        st["mode"] = "WAIT_STORAGE"; return 600
    seen, detail = alarm("10m")
    if seen:
        suspend_running(f"storage alarm back on: {detail}", ps); st["mode"] = "WAIT_STORAGE"; return 600
    for p in ps:
        n = p["metadata"]["name"]
        if p["status"].get("phase") != "Running": continue
        cs = (p["status"].get("containerStatuses") or [{}])[0].get("state", {}).get("running", {}).get("startedAt")
        if not cs: continue
        age = time.time() - datetime.datetime.strptime(cs, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()
        ps_ = st["pods"].setdefault(n, {})
        if not ps_.get("seen"):
            ps_["seen"] = True; event(f"POD-RUNNING {n} on {p['spec'].get('nodeName')} since {cs}")
        if age >= 900 and not ps_.get("check15"):
            path = save_log(n, False)
            if path:
                text = open(path).read(); g = gpu_vals(text)[-20:]
                keep = [l for l in text.splitlines() if re.search(r"^=== |fresh start|resume|ABORT|: OK$|already done|frames/s|^\[gpu\]", l)][-40:]
                top = kc("top", "pod", n, "-n", NS) or "kubectl top failed"
                open(os.path.join(D, f"check15-{n}.txt"), "w").write("\n".join(keep) + "\n" + top)
                ps_["check15"] = True
                event(f"CHECK15 {n}: last GPU samples {g}, mean {sum(g) / max(len(g), 1):.0f}%; {keep[-1] if keep else ''}")
        if age >= 1800:   # RUN stop condition: GPU under 40% over 10 minutes after the first 30
            o = kc("logs", n, "-n", NS, "--since=11m", timeout=120)
            g = gpu_vals(o or "")
            if len(g) >= 10 and sum(g) / len(g) < 40:
                ps_["idle"] = ps_.get("idle", 0) + 1
                status(f"idle strike {ps_['idle']} for {n}: mean {sum(g) / len(g):.0f}% over {len(g)} samples")
                if ps_["idle"] >= 2:
                    suspend_running(f"GPU of {n} under 40% for two checks (mean {sum(g) / len(g):.0f}%)", ps)
                    event("HOLD: idle GPU; Job stays suspended for the lead"); return None
            elif g: ps_["idle"] = 0
    return 300

if "--start-running" in sys.argv: st["mode"] = "RUNNING"; st["failed_base"] = 1
event(f"watcher start pid={os.getpid()} dry={DRY} mode={st['mode']}")
while time.time() < HARD_END:
    try:
        nap = wait_storage() if st["mode"] == "WAIT_STORAGE" else running()
    except Exception as e:
        status(f"loop error: {e!r}"[:400]); nap = 300
    if nap is None or ONCE: break
    time.sleep(nap)
event("watcher exit")
