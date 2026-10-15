# 05 — AI-Assisted and Agentic Work

Most of the work here is done by AI agents (Claude Code and subagents). These rules exist so we can move fast without losing track of what's true.

## Session protocol (every agent session)
**Start of session:**
1. Read `_standards/`, the direction's `STANDARDS.md`, its `README.md`, and the last few `LOG.md` entries.
2. Check `git status` and `git log -5` to see the current state.
3. Restate the goal for the session in one line. If it's unclear, ask.

**End of session:**
1. Append an entry to `LOG.md` (template in `04`). Mark the entry `[agent]` and include what was verified and what wasn't.
2. Update the `README.md` status and key findings if they changed.
3. Commit the work (see "Git" below).
4. Leave a concrete **Next** step, so the next session, human or agent, can start without re-deriving context.

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
- Anything that leaves the machine: pushing to a remote, publishing, emailing, uploading

## Working with subagents and parallel agents
- **Give subagents a scoped task** with a clear deliverable and pointers to the relevant standards files.
- **Treat subagent output as unverified** until the orchestrating agent or the human checks it, especially citations and numbers.
- **Use a separate branch or worktree** for parallel agents touching the same direction, so their work doesn't collide.

## Git
- **Commit often, in small units.** Use one commit per logical change, and commit before every experiment run.
- **Commit messages:** `<direction>: <what changed>`, for example `contrastive-pretraining: add linear-probe eval`.
- **Branch for risky or exploratory work:** `exp/<short-name>`. Merge into `main` once the work is understood.
- **Never commit** secrets, API keys, `.env` files, raw datasets, checkpoints, or large binaries (see `.gitignore`).

## For the human
- **Spot-check agent claims,** especially citations, surprising results, and anything going into a writeup.
- **Review `LOG.md` regularly.** The log is your audit trail of what the agents did.
- **Check the AI-use policy** of any course, lab, venue, or journal before submitting.
