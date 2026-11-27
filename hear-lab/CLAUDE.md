# hear-lab

This folder is the HEAR Lab direction (spatial audio) inside the research26-27 workspace. The workspace `CLAUDE.md` and `_standards/` apply here too, with this folder's `STANDARDS.md` on top. This file only says where the compute is and who may use it; the rules themselves live in the files it points to.

## Compute

- **Where:** Vast.ai first. Plan every run for Vast.ai unless the owner says Nautilus, or the experiment agent has explained why Nautilus fits a run better and the owner agreed. Nautilus is the Nautilus (NRP) cluster; its namespace is shared with the vis-lab work, so other sessions have Jobs running there too.
- **Who:** only the experiment agent (and workers it starts) rents or touches compute. Every other agent asks it instead.
- **Approval:** see "Rules for every agent" in the workspace `CLAUDE.md`. Before asking for a rental, state the machine (GPU, memory, disk, the host's reliability score), the hourly price, the expected hours and the total. Base the hours on a real timed batch.
- **Rules:** `STANDARDS.md`, "Where experiments run", and `../_standards/08-compute-nautilus.md` for any cluster work.
- **Vast.ai tool:** rent and manage instances with the `vastai` command line tool. If it is not installed or has no API key, stop and tell the owner so they can set it up. Never ask for the key in chat and never write it or any token into the repo. The saved key belongs to the HEAR Lab team account, which holds the lab's credits. Before renting, run `vastai show user` and check the username is HEAR Lab. If it shows the owner's personal account or asks for a two-factor login, stop and tell the owner; do not run `vastai tfa login`.
- **Vast.ai docs:** when unsure how something works, check https://docs.vast.ai/llms.txt (an index) or https://docs.vast.ai/llms-full.txt (everything in one large file; download it to your scratchpad and search it).
- **Vast.ai running:** a rented instance bills for as long as it exists. A few minutes into any new kind of run, check the GPU is busy. At the end of every session, tell the owner which instances still exist and what each costs per hour.
- **Nautilus:** submit and manage Jobs with kubectl. The nrp-accounting MCP server gives read-only usage for the namespace; it checks usage and manages nothing. When unsure, check the NRP docs (https://nrp.ai/llms.txt; download it to your scratchpad and search it). If the NRP docs and `_standards/08` disagree, follow `_standards` and tell the owner.
- **Cleanup:** whoever launches an instance or Job saves its logs and results, then destroys or deletes it.
