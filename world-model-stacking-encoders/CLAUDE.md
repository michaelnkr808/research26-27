# vis-lab (world-model-stacking-encoders)

This folder is the vis-lab research direction (stacking encoders) inside the research26-27 workspace. The workspace `CLAUDE.md` and `_standards/` apply here too, with this folder's `STANDARDS.md` on top. This file only says where the compute is and who may use it; the rules themselves live in the files it points to.

## Compute

- **Where:** the Nautilus (NRP) cluster, namespace `aria-world`. The namespace is shared with the rest of the lab, including the hear-lab work. The user is a member of it, not an admin. Leave alone anything you did not create.
- **Who:** only the experiment agent (and workers it starts) touches the cluster. Every other agent asks it instead.
- **Approval:** see "Rules for every agent" in the workspace `CLAUDE.md`. No new cluster Job is prepared or submitted, and nothing starts on paid compute, without the user's OK.
- **Rules:** `../_standards/08-compute-nautilus.md` and the "Compute budget" and "Paid compute" items of `STANDARDS.md`. When unsure how something on Nautilus works, check the NRP docs (https://nrp.ai/llms.txt; download it to your scratchpad and search it). If the NRP docs and `_standards/08` disagree, follow `_standards` and tell the user.
- **Tool:** submit and manage Jobs with kubectl.
- **Usage numbers:** the nrp-accounting MCP server gives read-only usage for the namespace (GPU, CPU, memory, by day and by node). It checks usage; it manages nothing.
- **Cost:** Nautilus is free, so there is no GPU-hour budget. The limits are the stop-and-ask gate in `STANDARDS.md` (more than 8 GPUs at once, or more than 20 Jobs) and NRP's utilization rules: requested GPUs stay above 40% use, and idle GPUs can get a pod deleted and a strike against the user (`_standards/08`). Paid compute is short jobs only, on the user's go with the price stated first (`STANDARDS.md`, "Paid compute").
- **Cleanup:** NRP does not clean up finished Jobs. Whoever launches a Job saves its logs and results, then deletes it. No interactive GPU pod is left running at the end of a session.
- **Ledger:** every pod's queue wait and run time goes in `compute/nautilus-job-times.csv`, read from kubectl or Prometheus, never estimated.
