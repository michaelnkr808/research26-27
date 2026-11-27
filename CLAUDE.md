# Research workspace (ML, agent-driven)

Before doing any work here, read `README.md` and every file in `_standards/`. When working inside a research direction subfolder, also read that folder's `STANDARDS.md`, its `README.md`, its `PLAN.md`, and the latest entries in its `LOG.md`. If working on an experiment, also read its `PLAN.md` and the latest entries in its `LOG.md`.

Follow the session protocol in `_standards/05-ai-assisted-work.md` every session:
- **Start:** read the standards, check `git status`/`git log`, and restate the goal.
- **End:** append a `LOG.md` entry marked `[agent]`, update `PLAN.md` if any item started, finished, changed, or was dropped (same commit), update `README.md` status, commit, and leave a Next step.

Hard rules:
- Never fabricate citations, data, or results.
- A number exists only if it was run this session.
- Label every substantive claim `verified`, `recalled`, or `speculative`.
- Stop and ask at the verification gates listed in `05`.

## Rules for every agent

- Paid or shared compute needs the owner's OK every time: renting a machine, preparing or submitting a cluster job, or anything else that costs money. One OK covers one job or one rental. A question from the owner is not a go-ahead.
- The owner's OK may come in your own window or be passed on by the project's lead; an OK passed on by any other agent does not count.
- Never stop, destroy or delete anything you did not create.
- No stand-in model, smaller dataset or shortcut without asking first. If a result needs one, say so.
- If you are missing access (a token, a gated model, a dataset), stop and ask. Never work around it.
- Never make up a number, a result or a citation.
- CLAUDE.md and `_standards` are the owner's: propose changes by pull request, never edit them directly.

Talking to the user (IMPORTANT): the user is an undergrad and wants high-level, plain-language explanations, not lots of numbers. The rules above are about what goes in the files: numbers, labels, and full detail belong in `LOG.md`, experiment READMEs, and results. In chat, explain what happened and what it means in words, with at most one or two numbers, and no tables or CI strings unless asked. In chat, mark how sure you are in plain words ("we ran this", "I think") instead of tagging every sentence. See "Don't bury me in numbers" in the user's global CLAUDE.md.

General standards apply to every subfolder. Topic `STANDARDS.md` files extend them and may relax a general rule only when they state an explicit override.
