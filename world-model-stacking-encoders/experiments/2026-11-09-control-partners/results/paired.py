"""Stage 1 paired differences from the fetched bootstrap draws (control partners, seed 0).

Run: /opt/homebrew/bin/python3.11 -I paired.py   (no network, no cluster; reads only files next to it).
Output: paired-report.txt (stdout, reproduced exactly on a rerun).

What the arrays are (from src/diagnostics.py, verified by reading it):
  bootstrap_r2 draws one (n_boot, n_test_demos) multinomial weight matrix from default_rng(seed) and
  returns, for each predictor, an (n_boot,) array of mean-over-dims test R2. `gain_draws[a|b|t]` is
  draws[min(a,b)+max(a,b)|t] - draws[a|t] = R2(a+b) - R2(a), one value per draw, so the array named
  "dinov3|X|global" is gain(X|DINOv3) per draw and "X|dinov3|global" is gain(DINOv3|X) per draw.
  The DINOv3-alone R2 draws themselves are NOT saved. ci() is np.quantile(draws, [0.025, 0.975])
  with NumPy's default linear interpolation; this script uses the same call.
"""
import base64
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LOG = HERE / "stacking-encoders-control-fetch-draws.log"
NPZDIR = HERE / "draws-s1"
CONTROLS = {  # control name -> (npz file, JSON file)
    "noise:4608": ("input-draws-noise-4608-s0.npz", "input-libero_goal-dinov3-noise-4608-s0.json"),
    "dup:dinov3": ("input-draws-dup-dinov3.npz", "input-libero_goal-dinov3-dup-dinov3.json"),
    "randproj:dinov3:4608": ("input-draws-randproj-dinov3-4608-s0.npz", "input-libero_goal-dinov3-randproj-dinov3-4608-s0.json"),
    "shuffle:dinov3": ("input-draws-shuffle-dinov3-s0.npz", "input-libero_goal-dinov3-shuffle-dinov3-s0.json"),
    "c-radiov4": ("input-draws-c-radiov4-rerun.npz", "input-libero_goal-dinov3-c-radiov4-rerun.json"),
}
N_BOOT, SEED, N_TEST_DEMOS = 1000, 0, 100
fails = []


def say(s=""):
    print(s)


def check(ok, what):
    say(f"  [{'pass' if ok else 'FAIL'}] {what}")
    if not ok:
        fails.append(what)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def ci(d, level=0.95):  # same as src/diagnostics.py ci()
    lo, hi = np.quantile(d, [(1 - level) / 2, (1 + level) / 2])
    return [float(lo), float(hi)]


# ---------------------------------------------------------------- 1. hashes
say("1. File integrity")
raw = LOG.read_bytes()
text = raw.decode("utf-8")
lines = text.splitlines()
say(f"  log {LOG.name}: {len(raw)} bytes, sha256 {sha(raw)}")
i0 = lines.index("=== sha256 of every file (regular files only)")
i1 = next(i for i in range(i0 + 1, len(lines)) if lines[i].startswith("=== "))
printed = {}
for ln in lines[i0 + 1:i1]:
    h, name = ln.split(None, 1)
    printed[name.strip().removeprefix("./")] = h
say(f"  volume listing: {len(printed)} files hashed in the log (5 npz, 5 input JSON, 5 target JSON)")
tar_hash_printed = next(re.match(r"([0-9a-f]{64})\s+/tmp/input-draws-s1\.tgz", l).group(1) for l in lines if re.match(r"[0-9a-f]{64}\s+/tmp/input-draws-s1\.tgz", l))
b0 = lines.index("=== BEGIN input-draws-s1.tgz base64")
b1 = lines.index("=== END input-draws-s1.tgz base64")
tgz = base64.b64decode("".join(lines[b0 + 1:b1]))
check(sha(tgz) == tar_hash_printed, f"tarball decoded from the log has the printed sha256 ({sha(tgz)[:12]}...)")
members = {}
with tarfile.open(fileobj=io.BytesIO(tgz), mode="r:gz") as tf:  # read in memory, nothing extracted
    for m in tf.getmembers():
        if m.isfile():
            members[m.name.removeprefix("./")] = tf.extractfile(m).read()
check(sorted(members) == sorted(n for n in printed if n.endswith(".npz")), "tarball holds exactly the 5 npz files on the volume")
for name, b in sorted(members.items()):
    check(sha(b) == printed[name], f"tar member {name} matches the volume's printed sha256")
    check((NPZDIR / name).read_bytes() == b, f"committed draws-s1/{name} is byte-identical to the tar member")
say("  committed result JSONs versus the volume's printed sha256 (trailing newline):")
n_nl = 0
for name in sorted(n for n in printed if n.endswith(".json")):
    b = (HERE / name).read_bytes()
    same_as_is = sha(b) == printed[name]
    minus_nl = b.endswith(b"\n") and sha(b[:-1]) == printed[name]
    n_nl += minus_nl
    check(minus_nl and not same_as_is, f"{name}: repo file minus its final newline hashes to the volume value; repo file with the newline does not")
say(f"  {n_nl} of {sum(n.endswith('.json') for n in printed)} JSONs differ from the volume copy only by one trailing newline")

# ---------------------------------------------------------------- load
draws, js = {}, {}
for c, (npz, jf) in CONTROLS.items():
    z = np.load(NPZDIR / npz)
    draws[c] = {k: z[k] for k in z.files}
    js[c] = json.loads((HERE / jf).read_text())
    expect = {f"dinov3|{c}|global", f"{c}|dinov3|global"}
    check(set(draws[c]) == expect and all(v.shape == (N_BOOT,) and v.dtype == np.float64 for v in draws[c].values()),
          f"{npz}: keys {sorted(draws[c])}, each float64 x {N_BOOT}")
    check(js[c]["n_boot"] == N_BOOT and js[c]["seed"] == SEED and js[c]["n_test_demos"] == N_TEST_DEMOS,
          f"{jf}: n_boot {js[c]['n_boot']}, seed {js[c]['seed']}, test demos {js[c]['n_test_demos']}")

# ---------------------------------------------------------------- 2. reproduce CIs
say()
say("2. Saved draws reproduce the CIs in the stage 1 JSONs (method: np.quantile at 0.025 and 0.975, linear interpolation, as src/diagnostics.py ci())")
tol = 1e-9
say(f"  {'control':22s} {'array':28s} {'JSON ci95':26s} {'from draws':26s} max abs diff")
for c in CONTROLS:
    for k, d in draws[c].items():
        a, b, t = k.split("|")
        pr = js[c]["pairs"][k]
        got = ci(d)
        diff = max(abs(got[0] - pr["gain_ci95"][0]), abs(got[1] - pr["gain_ci95"][1]))
        say(f"  {c:22s} {k:28s} [{pr['gain_ci95'][0]:+.6f}, {pr['gain_ci95'][1]:+.6f}] [{got[0]:+.6f}, {got[1]:+.6f}] {diff:.2e}")
        check(diff < tol, f"{c}: CI of {k} from draws equals the JSON gain_ci95 (diff {diff:.2e} < {tol:g})")
    # the JSON also holds asym_ci95 = CI of R2(X) - R2(D), which equals the CI of (gain(X|D) draws - gain(D|X) draws)
    g1 = draws[c][f"dinov3|{c}|global"]
    g2 = draws[c][f"{c}|dinov3|global"]
    ac = ci(g1 - g2)
    aj = js[c]["pairs"][f"dinov3|{c}|global"]["asym_ci95"]
    diff = max(abs(ac[0] - aj[0]), abs(ac[1] - aj[1]))
    check(diff < tol, f"{c}: CI of (gain(X|D) draws - gain(D|X) draws) = R2(X) - R2(D) equals the JSON asym_ci95 (diff {diff:.2e})")
    pt = js[c]["pairs"][f"dinov3|{c}|global"]["gain_B_given_A"]
    say(f"  {c:22s} mean of gain draws {g1.mean():+.6f} against point value {pt:+.6f}")

# ---------------------------------------------------------------- 3. pairing
say()
say("3. Are the draws paired across the five runs?")
say("  The PLAN's claim: the weights depend only on n_test_demos, n_boot and the seed. Code (bootstrap_r2):")
say("  w = np.random.default_rng(seed).multinomial(k, np.full(k, 1/k), size=n_boot), k = number of test demos.")
say(f"  All five JSONs record n_boot {N_BOOT}, seed {SEED}, {N_TEST_DEMOS} test demos (checked above), so the Job used one weight matrix for all five runs.")
w = np.random.default_rng(SEED).multinomial(N_TEST_DEMOS, np.full(N_TEST_DEMOS, 1.0 / N_TEST_DEMOS), size=N_BOOT).astype(np.float64)
say(f"  Reconstructed w: shape {w.shape}, every row sums to {int(w.sum(1).min())}-{int(w.sum(1).max())}, sha256 of its bytes {sha(w.tobytes())[:16]}...")
say("  The DINOv3-alone R2 draws are not in the npz files (only gain arrays are saved), so the exact-identity test of those")
say("  draws cannot be run. Three weaker tests that the files and JSONs do allow:")
say("  (a) DINOv3 alone is the same probe in every run, so its R2 point value and its bootstrap CI (computed from the DINOv3-alone")
say("      draws inside each run) must be the same in all five JSONs if the weights were the same:")
d0 = js["c-radiov4"]["probes"]["dinov3|global"]
mx = 0.0
for c in CONTROLS:
    pd_ = js[c]["probes"]["dinov3|global"]
    mx = max(mx, abs(pd_["r2_mean"] - d0["r2_mean"]), abs(pd_["r2_ci95"][0] - d0["r2_ci95"][0]), abs(pd_["r2_ci95"][1] - d0["r2_ci95"][1]))
say(f"      DINOv3 alone: R2 {d0['r2_mean']:.6f}, CI [{d0['r2_ci95'][0]:.6f}, {d0['r2_ci95'][1]:.6f}]; max abs difference over the five JSONs {mx:.3e}")
check(mx == 0.0, "DINOv3-alone R2 and CI are identical in all five JSONs (max abs difference exactly 0)")
say("  (b) suggestive, not a proof. q_X = gain(D|X) draws - gain(X|D) draws = R2(D) - R2(X) per draw. R2(D) has a bootstrap spread of its own")
say("      (sd of q_noise below, about 0.012) and R2(X) is near 0 for noise and shuffle (their JSONs). If two runs used different weights,")
say("      q_noise - q_shuffle would contain the difference of two independent R2(D) draws, with spread about sqrt(2) times sd(q). If the same")
say("      weights, that part cancels and only R2(shuffle) - R2(noise) per draw remains. The probes of X are not assumed to predict a constant.")
q = {c: draws[c][f"{c}|dinov3|global"] - draws[c][f"dinov3|{c}|global"] for c in CONTROLS}
for c in CONTROLS:
    say(f"      q_{c:22s} mean {q[c].mean():+.5f} sd {q[c].std(ddof=1):.5f}")
qn, qs = q["noise:4608"], q["shuffle:dinov3"]
r = np.corrcoef(qn, qs)[0, 1]
mad = np.abs(qn - qs).max()
ratio = (qn - qs).std(ddof=1) / (np.sqrt(2) * qn.std(ddof=1))
say(f"      noise:4608 vs shuffle:dinov3: corr {r:.8f}, max abs (q_noise - q_shuffle) {mad:.3e}, sd of the difference / (sqrt(2) * sd(q_noise)) = {ratio:.5f}")
perm = np.random.default_rng(12345).permutation(N_BOOT)
say(f"      reference for scale, one array with its draw order permuted (what unpaired looks like): corr {np.corrcoef(qn, qs[perm])[0, 1]:+.4f}")
check(r > 0.9999 and ratio < 0.01, "noise:4608 and shuffle:dinov3 runs: R2(D) fluctuation cancels in q_noise - q_shuffle (corr > 0.9999, difference spread under 1% of the unpaired value); suggestive of shared weights")
say("      randproj:dinov3:4608 and c-radiov4 are not used in (b): their own R2(X) is about R2(D) and fluctuates with the weights, so q_X does not isolate R2(D).")
say("  (c) dup:dinov3 is the same features as DINOv3, so its probe predictions equal DINOv3's and R2(D) - R2(X) must be 0 in every draw:")
dq = q["dup:dinov3"]
say(f"      dup:dinov3: max abs q {np.abs(dq).max():.3e}")
check(np.abs(dq).max() == 0.0, "dup:dinov3 R2(D) - R2(X) is exactly 0 in every draw (shows the two probes inside that run saw the same weights; not a cross-run test)")
say("      The thresholds in (b) (corr > 0.9999, ratio < 0.01) were chosen after seeing the data.")
say("  (d) Data-level test for all ten arrays. Rebuild the weights exactly as bootstrap_r2 does (one default_rng(0), one multinomial(100, 1/100, size=1000) call).")
say("      Each saved array is a smooth function of the weights of its own draw, so a linear regression of the array on the weight matrix, fitted on the first 700")
say("      draws and scored (out-of-sample R2) on the last 300, should predict well in generator order and not when the draw order is scrambled.")
say("      Criterion, written down before this script's own output was looked at but after the reviewer's numbers were known (in-order 0.942 to 0.988; best of 20")
say("      permutations -0.08 or worse; shifted -0.15): in-order score above 0.8, and every one of 20 permuted orders (permutation seeds 0..19) and the shift-by-one below 0.2.")
IN_MIN, PERM_MAX, N_PERM = 0.8, 0.2, 20
tr, te = np.arange(0, 700), np.arange(700, 1000)
XW = np.column_stack([np.ones(N_BOOT), w])


def oos_r2(y):
    beta = np.linalg.lstsq(XW[tr], y[tr], rcond=None)[0]
    res = y[te] - XW[te] @ beta
    return 1 - (res ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()


say(f"      {'control':22s} {'array':32s} {'in order':>9s} {'best of 20 permuted':>20s} {'shifted by 1':>13s}")
all_ok = True
for c in CONTROLS:
    for k, y in draws[c].items():
        r_in = oos_r2(y)
        r_perm = max(oos_r2(y[np.random.default_rng(i).permutation(N_BOOT)]) for i in range(N_PERM))
        r_sh = oos_r2(np.roll(y, 1))
        all_ok &= bool(r_in > IN_MIN and r_perm < PERM_MAX and r_sh < PERM_MAX)
        say(f"      {c:22s} {k:32s} {r_in:9.4f} {r_perm:20.4f} {r_sh:13.4f}")
check(all_ok, f"all ten arrays: in-order out-of-sample R2 above {IN_MIN} and permuted/shifted below {PERM_MAX}")
say("      Limit: this is statistical, not bit-exact per draw. Bit-exact identity rests on the code: one generator, one multinomial call, nothing partner-dependent")
say("      drawing from it, the same demo order (all five JSONs record the same frames, split, files and 100 test demos), and the same NumPy and source version in one pod.")
say("  With (a) to (d), pairing follows from the code and is confirmed at data level for all five files; (a) to (c) are the weaker earlier checks.")
if fails:
    say("PAIRING CHECK FAILED: no paired interval computed.")
    sys.exit(1)

# ---------------------------------------------------------------- 4. paired differences
say()
say("4. Paired difference gain(C-RADIOv4|DINOv3) minus gain(control|DINOv3), per draw (95% interval: same np.quantile method)")
R = "c-radiov4"
gR = draws[R][f"dinov3|{R}|global"]
pR = js[R]["pairs"][f"dinov3|{R}|global"]["gain_B_given_A"]
say(f"  gain(C-RADIOv4|DINOv3): point {pR:+.6f} (rerun JSON), CI from draws [{ci(gR)[0]:+.6f}, {ci(gR)[1]:+.6f}]")
say(f"  {'control':22s} {'gain(ctrl|D) point':>19s} {'paired diff point':>18s} {'95% interval from draws':>30s} {'mean of draws':>14s} {'share of draws > 0':>19s}")
res = {}
for c in ["noise:4608", "dup:dinov3", "randproj:dinov3:4608", "shuffle:dinov3"]:
    gc = draws[c][f"dinov3|{c}|global"]
    pc = js[c]["pairs"][f"dinov3|{c}|global"]["gain_B_given_A"]
    d = gR - gc
    lo, hi = ci(d)
    res[c] = (pc, pR - pc, lo, hi)
    say(f"  {c:22s} {pc:+19.6f} {pR - pc:+18.6f} {'[%+.6f, %+.6f]' % (lo, hi):>30s} {d.mean():+14.6f} {(d > 0).mean():19.3f}")
perm0 = np.random.default_rng(777).permutation(N_BOOT)
dsc = gR - draws["noise:4608"]["dinov3|noise:4608|global"][perm0]
say(f"  For scale only (not a result): scrambling the draw order of the noise array (permutation seed 777) gives an interval [{ci(dsc)[0]:+.6f}, {ci(dsc)[1]:+.6f}] against the paired [{res['noise:4608'][2]:+.6f}, {res['noise:4608'][3]:+.6f}].")
say()
lo = res["noise:4608"][2]
say(f"  PLAN note (Predictions, input side, 'below -0.005 with CI under 0'): R's +0.037 is information on top of a width penalty (paired difference larger than +0.037).")
say(f"  paired difference against noise:4608: point {res['noise:4608'][1]:+.6f}, interval [{res['noise:4608'][2]:+.6f}, {res['noise:4608'][3]:+.6f}];")
say(f"  gain(C-RADIOv4|DINOv3) point is {pR:+.6f}; the paired difference is larger than it: {res['noise:4608'][1] > pR} (point), interval lower end above it: {res['noise:4608'][2] > pR}.")
say()
say("Checks failed: " + (str(len(fails)) if fails else "none"))
if fails:
    for f in fails:
        say("  - " + f)
    sys.exit(1)

# ================================================================ 5. STAGE 2 (seeds 1 and 2), appended; nothing above was edited
say()
say("=" * 70)
say("STAGE 2: seeds 1 and 2 of the random controls (paired with the stage 1 C-RADIOv4 rerun draws)")
say("=" * 70)
LOG2 = HERE / "control-partners-s2-20261111.log"
NPZDIR2 = HERE / "draws-s2"
fails2 = []


def check2(ok, what):
    say(f"  [{'pass' if ok else 'FAIL'}] {what}")
    if not ok:
        fails2.append(what)


raw2 = LOG2.read_bytes()
lines2 = raw2.decode("utf-8").split("\n")
say("5.1 File integrity (stage 2)")
say(f"  log {LOG2.name}: {len(raw2)} bytes, sha256 {sha(raw2)}")
j0 = lines2.index("=== sha256 of every output file (regular files only)")
j1 = next(i for i in range(j0 + 1, len(lines2)) if lines2[i].startswith("=== "))
printed2 = {}
for ln in lines2[j0 + 1:j1]:
    if re.match(r"[0-9a-f]{64}  ", ln):
        h, nm = ln.split(None, 1)
        printed2[nm.strip().removeprefix("./")] = h
say(f"  volume listing: {len(printed2)} files hashed in the log (6 npz, 6 input JSON, 8 target JSON)")
tar2 = next(re.match(r"([0-9a-f]{64})\s+/work/input-draws-s2\.tgz", l).group(1) for l in lines2 if re.match(r"[0-9a-f]{64}\s+/work/input-draws-s2\.tgz", l))
c0 = lines2.index("=== BEGIN input-draws-s2.tgz base64")
c1 = lines2.index("=== END input-draws-s2.tgz base64")
tgz2 = base64.b64decode("".join(lines2[c0 + 1:c1]))
check2(sha(tgz2) == tar2, f"tarball decoded from the log has the printed sha256 ({sha(tgz2)[:12]}...)")
members2 = {}
with tarfile.open(fileobj=io.BytesIO(tgz2), mode="r:gz") as tf:
    for m in tf.getmembers():
        if m.isfile():
            members2[m.name.removeprefix("./")] = tf.extractfile(m).read()
check2(sorted(members2) == sorted(n for n in printed2 if n.endswith(".npz")), "tarball holds exactly the 6 npz files on the volume")
for name, b in sorted(members2.items()):
    check2(sha(b) == printed2[name], f"tar member {name} matches the volume's printed sha256")
    check2((NPZDIR2 / name).read_bytes() == b, f"committed draws-s2/{name} is byte-identical to the tar member")
say("  committed stage 2 result JSONs versus the volume's printed sha256 (trailing newline):")
n_nl2 = 0
for name in sorted(n for n in printed2 if n.endswith(".json")):
    b = (HERE / name).read_bytes()
    minus_nl = b.endswith(b"\n") and sha(b[:-1]) == printed2[name]
    n_nl2 += minus_nl
    check2(minus_nl and sha(b) != printed2[name], f"{name}: repo file minus its final newline hashes to the volume value")
say(f"  {n_nl2} of {sum(n.endswith('.json') for n in printed2)} JSONs differ from the volume copy only by one trailing newline")

S2C = {"noise:4608": "noise-4608", "randproj:dinov3:4608": "randproj-dinov3-4608", "shuffle:dinov3": "shuffle-dinov3"}
draws2, js2 = {}, {}
for sd in (1, 2):
    for c, stem in S2C.items():
        z = np.load(NPZDIR2 / f"input-draws-{stem}-s{sd}.npz")
        draws2[(c, sd)] = {k: z[k] for k in z.files}
        js2[(c, sd)] = json.loads((HERE / f"input-libero_goal-dinov3-{stem}-s{sd}.json").read_text())
        jj = js2[(c, sd)]
        expect = {f"dinov3|{c}|global", f"{c}|dinov3|global"}
        check2(set(draws2[(c, sd)]) == expect and all(v.shape == (N_BOOT,) and v.dtype == np.float64 for v in draws2[(c, sd)].values()),
               f"input-draws-{stem}-s{sd}.npz: keys {sorted(draws2[(c, sd)])}, each float64 x {N_BOOT}")
        check2(jj["n_boot"] == N_BOOT and jj["seed"] == SEED and jj["n_test_demos"] == N_TEST_DEMOS and jj["encoders"][c]["seed"] == sd,
               f"input JSON {c} seed {sd}: n_boot {jj['n_boot']}, bootstrap seed {jj['seed']}, test demos {jj['n_test_demos']}, synthetic seed {jj['encoders'][c]['seed']}")

say()
say("5.2 Saved draws reproduce the CIs in the stage 2 JSONs (np.quantile 0.025 / 0.975, as in section 2)")
say(f"  {'control':22s} seed {'array':34s} {'JSON ci95':26s} {'from draws':26s} max abs diff")
for sd in (1, 2):
    for c in S2C:
        for k, d in draws2[(c, sd)].items():
            pr = js2[(c, sd)]["pairs"][k]
            got = ci(d)
            diff = max(abs(got[0] - pr["gain_ci95"][0]), abs(got[1] - pr["gain_ci95"][1]))
            say(f"  {c:22s} {sd:4d} {k:34s} [{pr['gain_ci95'][0]:+.6f}, {pr['gain_ci95'][1]:+.6f}] [{got[0]:+.6f}, {got[1]:+.6f}] {diff:.2e}")
            check2(diff < tol, f"{c} seed {sd}: CI of {k} from draws equals the JSON gain_ci95 (diff {diff:.2e} < {tol:g})")
        g1 = draws2[(c, sd)][f"dinov3|{c}|global"]
        g2 = draws2[(c, sd)][f"{c}|dinov3|global"]
        ac = ci(g1 - g2)
        aj = js2[(c, sd)]["pairs"][f"dinov3|{c}|global"]["asym_ci95"]
        diff = max(abs(ac[0] - aj[0]), abs(ac[1] - aj[1]))
        check2(diff < tol, f"{c} seed {sd}: CI of (gain(X|D) draws - gain(D|X) draws) equals the JSON asym_ci95 (diff {diff:.2e})")
        pt = js2[(c, sd)]["pairs"][f"dinov3|{c}|global"]["gain_B_given_A"]
        say(f"  {c:22s} {sd:4d} mean of gain draws {g1.mean():+.6f} against point value {pt:+.6f}")

say()
say("5.3 Are the stage 2 draws paired with the stage 1 C-RADIOv4 rerun draws?")
say("  Same code path as section 3: bootstrap seed 0 and 100 test demos in every JSON (checked above and in sections 1 to 3), so one weight matrix, rebuilt above as w.")
say("  Weaker checks: the DINOv3-alone global probe R2 and its bootstrap CI across all 18 input runs (stage 1 and 2). Stage 2 ran on another node, so exact equality is not required of")
say("  the R2 value (floating point), but the CI is a function of the same weights:")
d_all = [js["c-radiov4"]["probes"]["dinov3|global"]] + [js[c]["probes"]["dinov3|global"] for c in CONTROLS] + [js2[k]["probes"]["dinov3|global"] for k in js2]
mx_r2 = max(abs(x["r2_mean"] - d0["r2_mean"]) for x in d_all)
mx_ci = max(max(abs(x["r2_ci95"][0] - d0["r2_ci95"][0]), abs(x["r2_ci95"][1] - d0["r2_ci95"][1])) for x in d_all)
say(f"  DINOv3 alone: stage 1 R2 {d0['r2_mean']:.9f}, CI [{d0['r2_ci95'][0]:.9f}, {d0['r2_ci95'][1]:.9f}]; max abs difference over all runs: R2 {mx_r2:.2e}, CI ends {mx_ci:.2e}")
check2(mx_r2 < 1e-6 and mx_ci < 1e-6, "DINOv3-alone R2 and CI agree to under 1e-6 across stage 1 and stage 2 runs (not required to be bit-equal: another node)")
say("  Data-level test (d) on the 12 stage 2 arrays: same regression of each saved array on the rebuilt weight matrix, same criterion as section 3 (in order above 0.8; 20 permuted orders and shift-by-one below 0.2):")
say(f"      {'control':22s} seed {'array':32s} {'in order':>9s} {'best of 20 permuted':>20s} {'shifted by 1':>13s}")
all_ok2 = True
for sd in (1, 2):
    for c in S2C:
        for k, y in draws2[(c, sd)].items():
            r_in = oos_r2(y)
            r_perm = max(oos_r2(y[np.random.default_rng(i).permutation(N_BOOT)]) for i in range(N_PERM))
            r_sh = oos_r2(np.roll(y, 1))
            all_ok2 &= bool(r_in > IN_MIN and r_perm < PERM_MAX and r_sh < PERM_MAX)
            say(f"      {c:22s} {sd:4d} {k:32s} {r_in:9.4f} {r_perm:20.4f} {r_sh:13.4f}")
check2(all_ok2, f"all 12 stage 2 arrays: in-order out-of-sample R2 above {IN_MIN} and permuted/shifted below {PERM_MAX}")
say("  Same limit as in section 3: statistical, not bit-exact per draw; bit-exact identity rests on the code (one generator and call, same demo order).")

say()
say("5.4 Paired difference gain(C-RADIOv4|DINOv3) minus gain(control, seed s|DINOv3), per draw; C-RADIOv4 draws are the stage 1 rerun's (a real encoder: one value for every seed)")
say(f"  gain(C-RADIOv4|DINOv3): point {pR:+.6f}, CI from draws [{ci(gR)[0]:+.6f}, {ci(gR)[1]:+.6f}]")
say(f"  {'control':22s} seed {'gain(ctrl|D) point':>19s} {'paired diff point':>18s} {'95% interval from draws':>30s} {'mean of draws':>14s} {'share of draws > 0':>19s}")
res2 = {}
for c in S2C:
    for sd in (0, 1, 2):
        if sd == 0:
            gc = draws[c][f"dinov3|{c}|global"]
            pc = js[c]["pairs"][f"dinov3|{c}|global"]["gain_B_given_A"]
        else:
            gc = draws2[(c, sd)][f"dinov3|{c}|global"]
            pc = js2[(c, sd)]["pairs"][f"dinov3|{c}|global"]["gain_B_given_A"]
        d = gR - gc
        lo, hi = ci(d)
        res2[(c, sd)] = (pc, pR - pc, lo, hi)
        say(f"  {c:22s} {sd:4d} {pc:+19.6f} {pR - pc:+18.6f} {'[%+.6f, %+.6f]' % (lo, hi):>30s} {d.mean():+14.6f} {(d > 0).mean():19.3f}")
say("  (seed 0 rows repeat section 4 for these three controls.)")
say(f"  PLAN note (input side): for noise below -0.005 with CI under 0, R's +0.037 is information on top of a width penalty (paired difference larger than +0.037). Interval lower end above {pR:+.6f}:")
for sd in (0, 1, 2):
    say(f"    noise:4608 seed {sd}: paired difference {res2[('noise:4608', sd)][1]:+.6f}, interval [{res2[('noise:4608', sd)][2]:+.6f}, {res2[('noise:4608', sd)][3]:+.6f}], lower end above gain(C-RADIOv4|DINOv3): {res2[('noise:4608', sd)][2] > pR}")
say("  As in stage 1: per draw this is the same statement as 'the noise control's gain is below 0'; the 'Capacity floor above zero' clause, the only clause that uses a paired difference, fires on no seed (collect-report.txt).")
say()
say("Stage 2 checks failed: " + (str(len(fails2)) if fails2 else "none"))
if fails2:
    for f in fails2:
        say("  - " + f)
    sys.exit(1)
