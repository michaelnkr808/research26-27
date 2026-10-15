# 04 — Documentation

## Research log (`LOG.md`)
- **Append-only and dated.** Add the newest entries at the bottom, and never rewrite old ones. Add a correction below the original instead.
- **Write an entry for every working session**, even if it's short:

```markdown
## 2026-10-15
**Goal:** what I set out to do
**Did:** what actually happened
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
Keep this file current. It should always answer: what is the question, what is the status, what do we know so far, and where are the key files.

## Ready to share
Work is mostly solo but may be shared, so write everything as if a collaborator could pick it up tomorrow:
- `README.md` and `LOG.md` should make sense without the conversation that produced them.
- Record ownership in the direction's `README.md` (who leads it, who contributes) once anyone else is involved.
- Keep secrets, credentials, and personal paths out of committed files. Use environment variables and a `.env.example`.
- Before sharing a direction, check that nothing in it depends on files outside the folder or the repo without documenting it.

## Naming
- Folders and files: `kebab-case`, with no spaces.
- Dates: `YYYY-MM-DD`, always.
- Version files with git, or with `-v2` / `-v3`. Never use `final`, `final2`, or `new`.
