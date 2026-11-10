# Paper draft (stacking encoders)

The LaTeX draft of the case-study paper for a CVPR workshop (target: the mentor's goal, user 2026-11-10). The plan behind it, with the outline, sources, slots per ending and the do-not-repeat list, is `../2026-11-10-writing-plan.md` (PR #12). Which workshop, its deadline, its length and whether it is archival are not decided yet (user and mentor; the CVPR 2027 workshop list is due after 2026-12-22).

## Who writes what (user's choice, 2026-11-10: "Mix by section")
- The user writes: abstract, introduction, related work, discussion, from the annotated outlines in those files. The draft session fact-checks them.
- The draft session drafts: setup, clean-frame tests, perturbed frames, proxy replication, the pair-measure slot if it runs, limitations, supplement. The user edits.
- Result slots are filled only after their milestone lands, using the wording of the locked rules in the experiment PLANs, and only after the user signs off the matching README finding (`../../../_standards/05`, verification gates).
- Unwritten slots print as red bold `[TODO: ...]` text in the PDF.

## Rules
- Changes land by pull request (`../../../_standards/05`, Git): this folder is a write-up, so every claim in it feeds a conclusion.
- Every sentence traces to a repo file or a cited paper; source pointers go in `%` comments next to the text.
- Cite only papers the user has read and that are in `../../literature/references.bib` (`../../../_standards/02`). The bib is read from there; there is no copy here.
- No acknowledgments in the review version (kit FAQ). `sec/ack.tex` holds the NRP text required by `../../../_standards/08` and is included only in the camera-ready version.
- Anonymity: the repo is public and holds this draft under the same title, so the review version must not link to it, and the chosen workshop's policy on public drafts needs checking.

## Build
In this folder: `tectonic main.tex` (Tectonic is installed via Homebrew on the Mac; it fetches LaTeX packages on first run). Output `main.pdf` is not committed. Tectonic runs XeTeX, so `main.tex` loads the TeX Gyre clones of Times, Helvetica and Courier there; without that the kit's Times setup falls back to Latin Modern. pdfLaTeX is not installed on the Mac, so the final page count is checked with pdfLaTeX, as the kit's CI uses (for example on Overleaf).

For Overleaf (for example a copy for the mentor): `main.tex` reads the bib as `../../literature/references`, which does not exist in an Overleaf project holding only this folder. Upload `references.bib` next to `main.tex` there and change that line in the Overleaf copy only; the repo keeps one bib.

## Before submission
- Workshop chosen: its kit (swap the three kit files and `\confName` if it has its own), page limit, deadline, review mode, AI-use policy (`../../../_standards/05`, "For the human").
- Every `[TODO]` resolved; then turn the macros off (the two commented lines in `preamble.tex`).
- Supplement out of the main PDF: `\withsupplfalse` in `main.tex`; submit `sec/X_suppl.tex` separately.
- Bib (known issues below) fixed; every reference shows an arXiv ID, DOI or stable URL; published versions cited where they exist.
- Page count checked with pdfLaTeX.
- Anonymity: no link to the repo, no names, no NRP text or acknowledgments in the review version.
- Camera-ready: switch `\usepackage[review]{cvpr}` to `\usepackage{cvpr}` (this brings in `sec/ack.tex`), fill the author list, OpenReview profiles for every author.

## Known issues in the bib
- The entries in `../../literature/references.bib` carry our bookkeeping in their `note` field (when the arXiv record was checked, whether the user read it), and the CVPR style prints `note` for `@misc` entries.
- The style has no `eprint` handling, so no reference prints its arXiv ID, while `../../../_standards/02` asks for a DOI, arXiv ID or stable URL on every citation.
- Fix before submission, not decided here: either the literature session moves the bookkeeping to a field the style ignores (for example `annote`) and adds a `url` or `howpublished` field, or the paper gets a cleaned copy of the bib. `references.bib` is shared and its entries are added by the literature session, so this folder does not edit it.

## Template provenance
`cvpr.sty`, `ieeenat_fullname.bst` and `preamble.tex` are copied unchanged from the official CVPR author kit, https://github.com/cvpr-org/author-kit, tag `CVPR2027-v1` (commit a2369342011145be8c85e5ad21a9b61f9fa53473), fetched 2026-11-10; the PR #13 reviewer re-fetched them and confirmed they are byte-identical. `main.tex` is adapted from the kit's `main.tex`; the `sec/` files are new. SHA-256 of the copied files:
- `cvpr.sty` ae8e51adc7270b118833b2092310ca1151a19fab0c09b1ab35a485ac4004ac0f
- `ieeenat_fullname.bst` e38e6166bd7b1e6d23a1b79dcdb55c656e4fcdbe91bdf6b50d827e6b5d1aacfc
- `preamble.tex` 080e0bae5ff30626292192ee2b4d1afca432ed2d2c939486c1bb37856166f8c1

The kit repo has no license file.
