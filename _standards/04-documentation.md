# 04 — Documentation

## Where things go
Each question about a direction has one file that answers it. The direction root stays high level; each experiment folder can be as detailed and technical as it needs.

**Direction root**
- `README.md`: the front page. The question, key findings, key files, a short status (one or two sentences), and a pointer to `PLAN.md`. No running history.
- `PLAN.md`: the living agenda. Only work that is running, next, or parked, under **Now / Next / Parked / Recently done**. No results. When an item finishes it becomes one line under "Recently done" pointing to its `LOG.md` entry; prune that list now and then. When an item is dropped or parked, record why (in the LOG entry or a decision record), not just that it moved.
- `LOG.md`: the direction's timeline. One short entry per session in plain words: what happened and what it means for the direction, pointing to the experiment `LOG.md` for detail. Work that belongs to no single experiment (decisions, literature, mentor feedback, plan changes) is logged here in full.

**Each experiment folder** (`experiments/YYYY-MM-DD-<short-name>/`, see `03`)
- `PLAN.md`: the design, written and committed **before** running. Question, design, and predictions or what would count as success. Short is fine, but every experiment has one. Once the experiment starts, the design is locked: changes are added as dated amendments below the original, never edits to it.
- `LOG.md`: the step-by-step record of running it. Jobs, errors, reruns, exact numbers.
- `README.md`: what came out. Results, where the outputs are, the commit hash, and links to the LOG entries. Written after (a stub saying "no results yet" is fine while it runs).

**There is no separate archive.** The direction `LOG.md`, the experiment `LOG.md`s and READMEs, and the README key findings together are the record.

**Plans written after the fact** must say so at the top ("Reconstructed YYYY-MM-DD from ..., not a new plan"), point to their source, and add no new design content.

## Logs (`LOG.md`, direction and experiment)
- **Append-only and dated.** Add the newest entries at the bottom, and never rewrite old ones. Add a correction below the original instead.
- **Write an entry for every working session**, even if it's short. A session that works on an experiment writes the detail in the experiment `LOG.md` and a short summary in the direction `LOG.md`:

```markdown
## 2026-10-15
**Goal:** what I set out to do
**Did:** what actually happened
**PR:** #N, or "direct"
**Found:** results/observations (link experiments/sources)
**Next:** concrete next step
**Open questions:**
```

## Decision records (`decisions/`)
Write a decision record for any choice that would be expensive to reverse, or that someone would later ask about, such as a method, dataset, metric, or scope cut.
`decisions/NNN-<short-title>.md`:

```markdown
# NNN — <Decision>
- **Date:**
- **Status:** proposed / accepted / superseded by NNN
- **Context:** what forced the decision
- **Options considered:**
- **Decision & rationale:**
- **Consequences:**
```

## Direction `README.md`
Keep this file current. It should always answer: what is the question, what is the status (briefly), what do we know so far, and where are the key files. What's being worked on goes in `PLAN.md`; how we got here goes in `LOG.md`.

## Ready to share
Work is mostly solo but may be shared, so write everything as if a collaborator could pick it up tomorrow:
- `README.md`, `PLAN.md`, and `LOG.md` should make sense without the conversation that produced them.
- Record ownership in the direction's `README.md` (who leads it, who contributes) once anyone else is involved.
- Keep secrets, credentials, and personal paths out of committed files. Use environment variables and a `.env.example`.
- Before sharing a direction, check that nothing in it depends on files outside the folder or the repo without documenting it.

## Naming
- Folders and files: `kebab-case`, with no spaces.
- Dates: `YYYY-MM-DD`, always.
- Version files with git, or with `-v2` / `-v3`. Never use `final`, `final2`, or `new`.
