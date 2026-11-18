"""Repair of damaged SigLIP 2 cache episodes (pairings stage 2): the steps around the re-extraction that the repair
Job (experiments/2026-11-15-pretrained-start-pairings/jobs/job-siglip2-repair.yaml) runs. Three subcommands:

  check-list  parse the damaged-episode list only (for the lead on the Mac, before the ConfigMap is made).
  prepare     parse the list, pre-check, snapshot, move the damaged files aside, write the extraction plan.
  postcheck   after the extractor ran: value check of every listed episode, the old _meta files unchanged, the new
              _meta records device cuda, nothing else in the suite folders changed; writes repair-record.json.

The list (results/cache-value-check/damaged-episodes.txt, written by RUN-cache-value-check.md step 6) has one line per
episode, "<source> <suite> <episode id>". Only "siglip2" lines are accepted; a line of another source, a malformed
line (not exactly three fields, an unknown suite, an id that is not a plain non-negative integer, an id not in
<feats>/meta/<suite>.npz), a blank line, a duplicate or an empty list is refused (exit 20) before anything is touched.

prepare, in this order, and nothing is changed before step 4:
  1. Parse the list (exit 20).
  2. State: <state> is the Job's own folder, /mnt/proxy/logs/siglip2-repair-<Job uid>. If <state>/snapshot.json exists,
     an earlier pod of this Job got past step 3; its list hash must match (else exit 22). Moved files in <state>
     without a snapshot: exit 22.
  3. Pre-check per listed suite: both _meta.shard0of2.json and _meta.shard1of2.json exist (exit 22) and, on a retry,
     still have the snapshot's sha256 (exit 25); their sha256 is printed. _meta.shard0of1.json must not exist, unless
     this Job's marker <state>/<suite>/meta-shard0of1.expected exists (written in step 5, before the extractor runs)
     (exit 22). Per listed episode: if this Job already moved a file of it (a copy in <state>/<suite>/), or a retry
     finds it in the snapshot with no file to move, it is left to the extractor and the post-check. Otherwise it must
     fail the value check now (check_stage2_cache.check_episode with values=True: header, stats, zero pages,
     non-finite; the same checks --values applies to one episode); a healthy one refuses the run (exit 21). A leftover
     temporary file (.ep<id>.npy.tmp, .ep<id>.stats.npy.tmp) of an episode not yet touched by this Job: exit 22.
  4. First pod only: <state> must be on the same file system as the suite folders (st_dev; else exit 23). Write
     <state>/snapshot.json (list hash, the episode files found, each with size and mtime, the full listing of every
     listed suite folder, the old _meta sha256, the start time), atomically. Then the moves: each original file
     (ep<id>.npy, ep<id>.stats.npy) of each listed episode is renamed (os.rename, never copy and delete) to
     <state>/<suite>/<name>. A file whose copy is already there is skipped; the file to move must still have the
     snapshot's size and mtime (exit 22). A rename error, cross-device (EXDEV) or other, exits 23. Nothing else is
     moved, deleted or written in the cache folders.
  5. Markers <state>/<suite>/meta-shard0of1.expected, then <state>/plan.txt: one line "<suite> <id>,<id>,..." per
     listed suite, for extract_fastwam.py --episodes (default --shard 0/1, so it writes _meta.shard0of1.json and
     leaves the two _meta.shard<i>of2.json files alone). Exit 0.

Why a retry is safe: the moves all happen before the extractor first runs, so while any original lacks its copy in
<state>, the extractor has not run and the file in the cache folder is the original; once every original has its
copy, nothing more is moved. The extractor skips an episode whose ep<id>.npy exists, so a repaired episode is not
recomputed, and one that was moved but not yet written is. State comes from the files only.

postcheck (exit codes 24 to 27; when several fail, the lowest is returned and every problem is printed):
  24  a listed episode still fails the value check (or is missing).
  25  an old _meta.shard<i>of2.json differs from its snapshot sha256 (before and after are printed).
  26  anything else in a listed suite folder changed against the snapshot listing (a name added or gone, a size or
      mtime changed), or a moved-aside copy is missing or differs from the size and mtime it had in the cache.
  27  the new _meta.shard0of1.json is missing, does not record device "cuda" or the given source commit, differs
      from the old _meta files in model, resolution or dtype, or check_stage2_cache.provenance reports a gap.
It writes <state>/repair-record.json either way (episodes, source commit, node, GPU name, start and end time, pod
starts, file counts, old _meta sha256, problems).

Run: python repair_stage2_episodes.py prepare --list /src/damaged-episodes.txt --feats /mnt/proxy/feats \
         --state /mnt/proxy/logs/siglip2-repair-<Job uid>
     python repair_stage2_episodes.py postcheck --list ... --feats ... --state ... --commit <sha> --gpu "<name>"
Other exit codes: 0 fine; 1 an uncaught exception (traceback; a pod retry resumes); 2 usage (argparse).
"""

import argparse
import errno
import hashlib
import json
import os
import re
import sys
import time

import check_stage2_cache as CS

SOURCE = "siglip2"
OLD_METAS = ("_meta.shard0of2.json", "_meta.shard1of2.json")
NEW_META = "_meta.shard0of1.json"
MARKER = "meta-shard0of1.expected"
SNAPSHOT = "snapshot.json"
PLAN = "plan.txt"
RECORD = "repair-record.json"
STARTS = "pod-starts.txt"

EXIT_LIST, EXIT_HEALTHY, EXIT_STATE, EXIT_MOVE = 20, 21, 22, 23
EXIT_STILL_DAMAGED, EXIT_META_CHANGED, EXIT_OTHER_CHANGED, EXIT_NEW_META = 24, 25, 26, 27


class Refuse(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def episode_files(ep):
    return [f"ep{ep:06d}.npy", f"ep{ep:06d}.stats.npy"]


def tmp_files(ep):
    return [f".ep{ep:06d}.npy.tmp", f".ep{ep:06d}.stats.npy.tmp"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_atomic(path, text):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(os.path.dirname(path))


def parse_list(text, suites=CS.SUITES):
    """[(suite, ep)] in list order, or Refuse(EXIT_LIST). Only siglip2 lines; no blank line, no duplicate."""
    out, bad, seen = [], [], set()
    for k, line in enumerate(text.splitlines(), 1):
        f = line.split()
        if len(f) != 3:
            bad.append(f"line {k}: {line!r}: want '<source> <suite> <episode id>'")
            continue
        src, suite, ep = f
        if src != SOURCE:
            bad.append(f"line {k}: {line!r}: source {src!r}, this repair takes only {SOURCE!r}")
            continue
        if suite not in suites:
            bad.append(f"line {k}: {line!r}: unknown suite {suite!r}")
            continue
        if not re.fullmatch(r"[0-9]+", ep):
            bad.append(f"line {k}: {line!r}: episode id {ep!r} is not a non-negative integer")
            continue
        key = (suite, int(ep))
        if key in seen:
            bad.append(f"line {k}: {line!r}: duplicate")
            continue
        seen.add(key)
        out.append(key)
    if bad:
        raise Refuse(EXIT_LIST, "damaged-episode list refused:\n  " + "\n  ".join(bad))
    if not out:
        raise Refuse(EXIT_LIST, "damaged-episode list is empty")
    return out


def by_suite(eps):
    d = {}
    for s, e in eps:
        d.setdefault(s, []).append(e)
    return d


def read_list(path):
    with open(path) as f:
        text = f.read()
    return text, hashlib.sha256(text.encode()).hexdigest()


def lengths(feats, suites):
    """{suite: {ep: length}} from <feats>/meta/<suite>.npz (check_stage2_cache.read_meta)."""
    out = {}
    for s in suites:
        m = CS.read_meta(feats, s)
        if m is None:
            raise Refuse(EXIT_STATE, f"{feats}/meta/{s}.npz missing")
        out[s] = dict(m)
    return out


def episode_gaps(feats, suite, ep, n, dims):
    """The value check of one episode, as check_stage2_cache.py --values does it (empty list: healthy)."""
    return CS.check_episode(feats, SOURCE, suite, ep, n, dims, values=True, tally=CS.new_tally())


def listing(folder):
    """{name: [size, mtime_ns, is_regular_file]} of every entry in folder (not recursive, symlinks not followed)."""
    out = {}
    with os.scandir(folder) as it:
        for e in it:
            st = e.stat(follow_symlinks=False)
            out[e.name] = [st.st_size, st.st_mtime_ns, e.is_file(follow_symlinks=False)]
    return out


def suite_dir(feats, suite):
    return os.path.join(feats, SOURCE, suite)


def load_snapshot(state):
    p = os.path.join(state, SNAPSHOT)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def prepare(list_path, feats, state, node="", dims=CS.DIMS, out=print):
    text, list_sha = read_list(list_path)
    eps = parse_list(text)
    groups = by_suite(eps)
    lens = lengths(feats, groups)
    unknown = [f"{s} {e}" for s, e in eps if e not in lens[s]]
    if unknown:
        raise Refuse(EXIT_LIST, f"episode(s) not in <feats>/meta/<suite>.npz: {', '.join(unknown)}")
    out(f"list: {len(eps)} episode(s), sha256 {list_sha}: " + "; ".join(f"{s} {sorted(v)}" for s, v in groups.items()))

    snap = load_snapshot(state)
    if snap is not None:
        if snap["list_sha256"] != list_sha:
            raise Refuse(EXIT_STATE, f"{state}/{SNAPSHOT} was made from another list (sha256 {snap['list_sha256']})")
        out(f"resume: {state}/{SNAPSHOT} exists (written {snap['created']}); an earlier pod of this Job got past the pre-check")
    else:
        moved = [os.path.join(s, n) for s in groups if os.path.isdir(os.path.join(state, s))
                 for n in os.listdir(os.path.join(state, s))]
        if moved:
            raise Refuse(EXIT_STATE, f"{state} holds {moved[:5]} but no {SNAPSHOT}")
        out(f"first pod: no {SNAPSHOT} in {state}")
    os.makedirs(state, exist_ok=True)
    with open(os.path.join(state, STARTS), "a") as f:
        f.write(f"{utc()} prepare node={node}\n")

    # pre-check: nothing is changed in this block
    old_sha, healthy = {}, []
    for s, ids in groups.items():
        sd = suite_dir(feats, s)
        if not os.path.isdir(sd):
            raise Refuse(EXIT_STATE, f"{sd} is not a folder")
        old_sha[s] = {}
        for m in OLD_METAS:
            p = os.path.join(sd, m)
            if not os.path.isfile(p):
                raise Refuse(EXIT_STATE, f"{p} missing: the repair must leave both shard-of-2 _meta files as they are")
            old_sha[s][m] = sha256_file(p)
            out(f"before: sha256 {old_sha[s][m]}  {SOURCE}/{s}/{m}")
            if snap is not None and snap["suites"][s]["old_meta_sha256"][m] != old_sha[s][m]:
                raise Refuse(EXIT_META_CHANGED, f"{p} changed since the snapshot "
                                                f"({snap['suites'][s]['old_meta_sha256'][m]})")
        if os.path.exists(os.path.join(sd, NEW_META)) and not os.path.exists(os.path.join(state, s, MARKER)):
            raise Refuse(EXIT_STATE, f"{sd}/{NEW_META} exists and was not written by this Job "
                                     f"(no {state}/{s}/{MARKER}); a repair or extraction ran here before")
        for e in ids:
            aside = [n for n in episode_files(e) if os.path.exists(os.path.join(state, s, n))]
            in_snap = snap is not None and str(e) in snap["originals"][s]
            if aside or (in_snap and not snap["originals"][s][str(e)]):
                have = [n for n in episode_files(e) if os.path.exists(os.path.join(sd, n))]
                out(f"  {s} ep{e:06d}: touched by this Job before (moved aside: {aside or 'nothing to move'}; "
                    f"in the cache now: {have or 'nothing'}); left to the extractor and the post-check")
                continue
            tmps = [n for n in tmp_files(e) if os.path.exists(os.path.join(sd, n))]
            if tmps:
                raise Refuse(EXIT_STATE, f"{sd}: leftover temporary file(s) {tmps} of an episode this Job has not touched")
            g = episode_gaps(feats, s, e, lens[s][e], dims)
            if not g:
                healthy.append(f"{s} {e}")
                out(f"  {s} ep{e:06d}: HEALTHY (passes the value check)")
            else:
                out(f"  {s} ep{e:06d}: damaged, {len(g)} gap(s): {g[0]}" + (f" (+{len(g) - 1} more)" if len(g) > 1 else ""))
    if healthy:
        raise Refuse(EXIT_HEALTHY, f"listed episode(s) pass the value check and were not moved by this Job: "
                                   f"{', '.join(healthy)}; nothing was moved")

    if snap is None:
        for s in groups:
            os.makedirs(os.path.join(state, s), exist_ok=True)
            if os.stat(suite_dir(feats, s)).st_dev != os.stat(os.path.join(state, s)).st_dev:
                raise Refuse(EXIT_MOVE, f"{state}/{s} is on another file system than {suite_dir(feats, s)}: "
                                        f"a rename cannot move the files; nothing was moved")
        snap = {"created": utc(), "list_sha256": list_sha, "episodes": [[s, e] for s, e in eps], "node": node,
                "suites": {}, "originals": {}}
        for s, ids in groups.items():
            lst = listing(suite_dir(feats, s))
            snap["suites"][s] = {"listing": lst, "files": len(lst), "old_meta_sha256": old_sha[s],
                                 "new_meta_existed": NEW_META in lst}
            snap["originals"][s] = {str(e): {n: lst[n][:2] for n in episode_files(e) if n in lst} for e in ids}
        write_atomic(os.path.join(state, SNAPSHOT), json.dumps(snap, indent=1, sort_keys=True))
        out(f"snapshot written: {state}/{SNAPSHOT}; files per suite: "
            + ", ".join(f"{s} {v['files']}" for s, v in snap["suites"].items()))

    # moves: rename only; each original file once
    nmoved = 0
    for s, ids in groups.items():
        sd, ad = suite_dir(feats, s), os.path.join(state, s)
        os.makedirs(ad, exist_ok=True)
        for e in ids:
            for n, (size, mtime) in snap["originals"][s][str(e)].items():
                src, dst = os.path.join(sd, n), os.path.join(ad, n)
                if os.path.exists(dst):
                    continue
                if not os.path.exists(src):
                    raise Refuse(EXIT_STATE, f"{src} is in the snapshot but neither in the cache nor in {ad}")
                st = os.stat(src)
                if [st.st_size, st.st_mtime_ns] != [size, mtime]:
                    raise Refuse(EXIT_STATE, f"{src} is not the file in the snapshot (size, mtime {st.st_size}, "
                                             f"{st.st_mtime_ns}; snapshot {size}, {mtime}); not moved")
                try:
                    os.rename(src, dst)
                except OSError as ex:
                    why = "cross-device rename (EXDEV)" if ex.errno == errno.EXDEV else f"{type(ex).__name__}: {ex}"
                    raise Refuse(EXIT_MOVE, f"rename {src} -> {dst} failed: {why}; no copy was made")
                nmoved += 1
                out(f"  moved aside: {src} -> {dst}")
        fsync_dir(sd)
        fsync_dir(ad)
    out(f"moves: {nmoved} file(s) renamed by this pod")

    for s in groups:
        p = os.path.join(state, s, MARKER)
        if not os.path.exists(p):
            write_atomic(p, f"{utc()} the extractor may write {SOURCE}/{s}/{NEW_META} from here on\n")
    plan = "".join(f"{s} {','.join(str(e) for e in ids)}\n" for s, ids in groups.items())
    write_atomic(os.path.join(state, PLAN), plan)
    out("plan:\n" + plan.rstrip())
    out("=== prepare OK")
    return 0


def postcheck(list_path, feats, state, commit="", node="", gpu="", dims=CS.DIMS, out=print):
    text, list_sha = read_list(list_path)
    eps = parse_list(text)
    groups = by_suite(eps)
    snap = load_snapshot(state)
    if snap is None:
        raise Refuse(EXIT_STATE, f"no {state}/{SNAPSHOT}: prepare did not run for this Job")
    if snap["list_sha256"] != list_sha:
        raise Refuse(EXIT_STATE, f"{state}/{SNAPSHOT} was made from another list")
    lens = lengths(feats, groups)
    probs = {}                                                    # code -> [messages]
    bad = lambda code, m: probs.setdefault(code, []).append(m)
    counts, metas = {}, {}
    for s, ids in groups.items():
        sd, ad = suite_dir(feats, s), os.path.join(state, s)
        for e in ids:
            g = episode_gaps(feats, s, e, lens[s][e], dims)
            out(f"  {s} ep{e:06d}: " + ("passes the value check" if not g else f"STILL DAMAGED: {'; '.join(g)}"))
            if g:
                bad(EXIT_STILL_DAMAGED, f"{s} ep{e:06d}: {g[0]}")
        old = {}
        for m in OLD_METAS:
            p = os.path.join(sd, m)
            h = sha256_file(p) if os.path.isfile(p) else None
            out(f"after: sha256 {h}  {SOURCE}/{s}/{m}")
            if h != snap["suites"][s]["old_meta_sha256"][m]:
                bad(EXIT_META_CHANGED, f"{p}: sha256 {h}, before {snap['suites'][s]['old_meta_sha256'][m]}")
            if h is not None:
                with open(p) as f:
                    old[m] = json.load(f)
        p = os.path.join(sd, NEW_META)
        if not os.path.isfile(p):
            bad(EXIT_NEW_META, f"{p} missing")
        else:
            with open(p) as f:
                j = json.load(f)
            metas[s] = {k: j.get(k) for k in ("model", "resolution", "dtype", "device", "git_commit")}
            out(f"new meta {SOURCE}/{s}/{NEW_META}: {metas[s]}")
            if j.get("device") != "cuda":
                bad(EXIT_NEW_META, f"{p}: device {j.get('device')!r}, want 'cuda'")
            if commit and j.get("git_commit") != commit:
                bad(EXIT_NEW_META, f"{p}: git_commit {j.get('git_commit')!r}, want {commit!r}")
            for m, o in old.items():
                for k in ("model", "resolution", "dtype"):
                    if j.get(k) != o.get(k):
                        bad(EXIT_NEW_META, f"{p}: {k} {j.get(k)!r}, {m} has {o.get(k)!r}")
        pg, plines = CS.provenance(feats, SOURCE, s)
        for line in plines:
            out(f"check_stage2_cache.provenance {s}:{line}")
        for x in pg:
            bad(EXIT_NEW_META, f"check_stage2_cache.provenance: {x}")

        before = snap["suites"][s]["listing"]
        now = listing(sd)
        mine = {n for e in ids for n in episode_files(e)}
        want = set(before) | mine | {NEW_META}
        counts[s] = [len(before), len(now)]
        out(f"files in {SOURCE}/{s}: before {len(before)}, after {len(now)} "
            f"(expected after {len(want)}: the same names, the repaired episodes' files and {NEW_META})")
        for n in sorted(set(now) - want):
            bad(EXIT_OTHER_CHANGED, f"{sd}/{n}: new file, not part of the repair")
        for n in sorted(want - set(now) - mine):
            bad(EXIT_OTHER_CHANGED, f"{sd}/{n}: gone")
        for n in sorted(set(before) & set(now) - mine - {NEW_META}):
            if before[n] != now[n]:
                bad(EXIT_OTHER_CHANGED, f"{sd}/{n}: size, mtime or type changed ({before[n]} -> {now[n]})")
        for e in ids:
            for n, ref in snap["originals"][s][str(e)].items():
                q = os.path.join(ad, n)
                if not os.path.exists(q):
                    bad(EXIT_OTHER_CHANGED, f"{q}: moved-aside copy missing")
                else:
                    st = os.stat(q)
                    if [st.st_size, st.st_mtime_ns] != ref:
                        bad(EXIT_OTHER_CHANGED, f"{q}: size or mtime differs from the file moved out of the cache")
    starts = []
    sp = os.path.join(state, STARTS)
    if os.path.exists(sp):
        with open(sp) as f:
            starts = f.read().splitlines()
    rc = min(probs) if probs else 0
    record = {"ok": rc == 0, "exit": rc, "source": SOURCE, "episodes": [[s, e] for s, e in eps],
              "list_sha256": list_sha, "source_commit": commit, "node": node, "gpu": gpu,
              "start": snap["created"], "end": utc(), "pod_starts": starts, "moved_aside_to": state,
              "files_before_after": counts, "old_meta_sha256": {s: snap["suites"][s]["old_meta_sha256"] for s in groups},
              "new_meta": metas, "problems": {str(k): v for k, v in sorted(probs.items())}}
    write_atomic(os.path.join(state, RECORD), json.dumps(record, indent=1, sort_keys=True))
    out(f"record: {state}/{RECORD}\n" + json.dumps(record, sort_keys=True))
    for k, v in sorted(probs.items()):
        for m in v:
            out(f"  PROBLEM (exit {k}) {m}")
    out("=== repair post-check " + ("OK" if rc == 0 else f"FAIL (exit {rc})"))
    return rc


def main(argv=None, out=print):
    p = argparse.ArgumentParser(description="Repair of damaged SigLIP 2 cache episodes: list check, pre-check and "
                                            "move aside, post-check (see the module docstring for exit codes).")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-list", help="parse the damaged-episode list only (exit 0 or 20)")
    c.add_argument("--list", required=True)
    for name, h in (("prepare", "pre-check, snapshot, move the damaged files aside, write the plan"),
                    ("postcheck", "value check of the repaired episodes, nothing else changed; repair-record.json")):
        q = sub.add_parser(name, help=h)
        q.add_argument("--list", required=True, help="damaged-episode list, lines '<source> <suite> <episode id>'")
        q.add_argument("--feats", default="/mnt/proxy/feats")
        q.add_argument("--state", required=True, help="this Job's folder, /mnt/proxy/logs/siglip2-repair-<Job uid>")
        q.add_argument("--node", default=os.environ.get("NODE_NAME", ""))
        if name == "postcheck":
            q.add_argument("--commit", default=os.environ.get("SRC_GIT_COMMIT", ""))
            q.add_argument("--gpu", default="")
    a = p.parse_args(argv)
    try:
        if a.cmd == "check-list":
            eps = parse_list(read_list(a.list)[0])
            for s, ids in by_suite(eps).items():
                out(f"{s} {','.join(str(e) for e in ids)}")
            out(f"=== list OK: {len(eps)} episode(s)")
            return 0
        if a.cmd == "prepare":
            return prepare(a.list, a.feats, a.state, a.node, out=out)
        return postcheck(a.list, a.feats, a.state, a.commit, a.node, a.gpu, out=out)
    except Refuse as r:
        out(f"REFUSE (exit {r.code}): {r}")
        return r.code


if __name__ == "__main__":
    sys.exit(main())
