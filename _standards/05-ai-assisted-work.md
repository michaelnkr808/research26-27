# 05 — AI-Assisted and Agentic Work

Most of the work here is done by AI agents (Claude Code and subagents). These rules exist so we can move fast without losing track of what's true.

## Session protocol (every agent session)
**Start of session:**
1. Read `_standards/`, the direction's `STANDARDS.md`, `README.md`, `PLAN.md`, and the last few `LOG.md` entries. If working on an experiment, also read its `PLAN.md` and the last few entries of its `LOG.md`.
2. Check `git status` and `git log -5` to see the current state.
3. Restate the goal for the session in one line. If it's unclear, ask.

**End of session:**
1. Append an entry to `LOG.md` (template in `04`). Mark the entry `[agent]` and include what was verified and what wasn't. If the session worked on an experiment, the detail goes in that experiment's `LOG.md` and a short summary in the direction's.
2. Update `PLAN.md` if any item started, finished, changed, or was dropped, **in the same commit** as the LOG entry (drift guard). For work that goes by PR, the PLAN status move goes in the same direct-merge commit as the LOG entry; design or outcome-rule changes were already in the PR. An open PR is a Now item until merged. A finished item becomes one line under "Recently done"; a dropped one keeps its reason in the LOG or a decision record.
3. Update the `README.md` status and key findings if they changed. If an experiment produced results, write them in its `README.md`.
4. Commit the work (see "Git" below).
5. Leave a concrete **Next** step, so the next session, human or agent, can start without re-deriving context.

## Truthfulness rules
- **Never fabricate** citations, data, metrics, quotes, library APIs, or results. If something is unknown, say so.
- **A result exists only once it has been run.** Agents must not report a number, or say "this works", unless they ran it in this session. Expected or predicted outcomes are labeled as such.
- **Label confidence** on every substantive claim:
  - `verified`: ran it, or checked it against a primary source
  - `recalled`: from memory or training data, not checked
  - `speculative`: a hypothesis or guess
- **Report failures plainly.** Say what was run, what failed (with the error), and what was skipped. Don't smooth failures into a summary.
- **Don't move goalposts silently.** If an agent changes a metric, split, baseline, or scope to make something work, it must log that change and flag it.

## Verification gates (human checks before proceeding)
An agent pauses and asks before:
- Treating a result as a **finding**, meaning it goes into `README.md` key findings or into any writeup
- Large compute jobs, beyond the budget set in the direction's `STANDARDS.md`
- Deleting data, overwriting results, rewriting git history, or force-pushing
- Changing the research question, scope, or evaluation protocol
- Anything that leaves the machine: pushing to a remote, publishing, emailing, uploading. Exceptions: pushing a task branch to open a pull request (and later fixes on it), and pushing `main` right after a merge (see Git).

## Working with subagents and parallel agents
- **Give subagents a scoped task** with a clear deliverable and pointers to the relevant standards files.
- **Treat subagent output as unverified** until the orchestrating agent or the human checks it, especially citations and numbers.
- **Use a separate branch or worktree** for parallel agents touching the same direction, so their work doesn't collide.
- **Several sessions share this repo.** Each lands its work through its own PR or direct merge (see Git). A session never merges another session's branch and never commits on `main`.

## Git
- **Commit often, in small units.** Use one commit per logical change, and commit before every experiment run.
- **Commit messages:** `<direction>: <what changed>`, for example `contrastive-pretraining: add linear-probe eval`.
- **Branch for every task.** Workers and parallel agents work on a branch or worktree, never on `main`. Name it `<direction>-<short-name>`, `standards-<short-name>` for root-level changes, or `exp/<short-name>` for risky or exploratory work.
- **`main` moves only by pull request** for any change that feeds a research conclusion: experiment code, analysis scripts, outcome rules and thresholds, decision records (`decisions/`), pre-committed memos in `writing/`, review reports in `reviews/`, experiment `PLAN.md` amendments, edits to `_standards/` or a direction `STANDARDS.md`, and anything that changes `README.md` key findings.
- **Direct merges to `main` (no PR) only for** `LOG.md` entries, `PLAN.md` Now/Next/Parked/Recently-done moves, `README.md` status lines, reading-list leads, and wording or typo fixes that change no fact. When in doubt, open a PR. Nobody makes a non-merge commit on `main`; a direct-merge chore is committed on a short-lived branch first.
- **The PR is the paper trail.** Its description carries the task as it was given; a fresh read-only reviewer posts findings as a PR comment; fixes land as new commits on the same branch; the `LOG.md` entry for the work names the PR number. The direction `LOG.md` entry is not committed on the task branch: it merges directly once the PR number exists. Experiment `LOG.md` entries ride with the work on the task branch.
- **Merging a PR needs the user's OK in chat,** in the session that opened it. GitHub cannot record an approval from the same account, so reviewer comment plus the user's OK is the approval. Merge on GitHub with a merge commit (`gh pr merge --merge --delete-branch`; no squash, no rebase), then `git pull --ff-only` local `main` before anything else.
- **After any merge to local `main`,** PR or direct, the merging session fast-forward pushes `main` to the remote in the same step.
- **The repo is public.** A pushed branch is visible at once, so it must satisfy "Never commit" (below) before it is pushed.
- **Never commit** secrets, API keys, `.env` files, raw datasets, checkpoints, or large binaries (see `.gitignore`).

## For the human
- **Spot-check agent claims,** especially citations, surprising results, and anything going into a writeup.
- **Review `LOG.md` regularly.** The log is your audit trail of what the agents did.
- **Check the AI-use policy** of any course, lab, venue, or journal before submitting.
