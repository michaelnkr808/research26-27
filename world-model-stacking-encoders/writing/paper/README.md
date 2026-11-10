# Paper draft (stacking encoders)

The LaTeX draft of the case-study paper for a CVPR workshop (target: the mentor's goal, user 2026-11-10). The plan behind it, with the outline, sources and slots per ending, is `../2026-11-10-writing-plan.md`. Which workshop, its deadline, page limit and whether it is archival are not decided yet (user and mentor).

## Who writes what (user's choice, 2026-11-10: "mix by section")
- The user writes: abstract, introduction, related work, discussion, from the annotated outlines in those files. The draft session fact-checks them.
- The draft session drafts: setup, clean-frame tests, perturbed frames, proxy replication, limitations, supplement. The user edits.
- Result slots are filled only after their milestone lands, using the wording of the locked rules in the experiment PLANs, and only after the user signs off the matching README finding (`../../../_standards/05`, verification gates).
- Red `[TODO: ...]` boxes in the PDF are unwritten slots.

## Rules
- Changes land by pull request (`../../../_standards/05`, Git): this folder is a write-up, so every claim in it feeds a conclusion.
- Every sentence traces to a repo file or a cited paper; source pointers go in `%` comments next to the text.
- Cite only papers the user has read and that are in `../../literature/references.bib` (`../../../_standards/02`). The bib is read from there; there is no copy here.
- No acknowledgments in the review version (kit FAQ). `sec/ack.tex` holds the NRP text required by `../../../_standards/08` and is included only in the camera-ready version.
- Anonymity: the repo is public, so the review version must not link to it.

## Build
In this folder: `tectonic main.tex` (Tectonic is installed via Homebrew on the Mac; it fetches LaTeX packages on first run). Output `main.pdf` is not committed.

## Known issue: bib notes print in the references
The entries in `../../literature/references.bib` carry our bookkeeping in their `note` field (when the arXiv record was checked, whether the user read it), and the CVPR style prints `note`. That text must not appear in a submitted paper. Fix before submission, not decided here: either the literature session moves that bookkeeping to a field the style ignores (for example `annote`), or the paper gets a cleaned copy of the bib. `references.bib` is shared and its entries are added by the literature session, so this folder does not edit it.

## Template provenance
`cvpr.sty`, `ieeenat_fullname.bst` and `preamble.tex` are copied unchanged from the official CVPR author kit, https://github.com/cvpr-org/author-kit, tag `CVPR2027-v1` (commit a2369342011145be8c85e5ad21a9b61f9fa53473), fetched 2026-11-10. `main.tex` and `sec/` are adapted from the kit's `main.tex`. SHA-256 of the copied files:
- `cvpr.sty` ae8e51adc7270b118833b2092310ca1151a19fab0c09b1ab35a485ac4004ac0f
- `ieeenat_fullname.bst` e38e6166bd7b1e6d23a1b79dcdb55c656e4fcdbe91bdf6b50d827e6b5d1aacfc
- `preamble.tex` 080e0bae5ff30626292192ee2b4d1afca432ed2d2c939486c1bb37856166f8c1

The kit repo has no license file. If the chosen workshop has its own kit, swap these three files and `\confName` in `main.tex`.
