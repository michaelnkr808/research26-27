# 02 — Sources and Citations

## Trust tiers (prefer higher tiers)
1. **Primary**: peer-reviewed papers, official specs and standards, original datasets, official documentation, source code.
2. **Strong secondary**: preprints from credible groups (arXiv etc., flagged as *not peer reviewed*), textbooks, survey papers, technical reports.
3. **Weak secondary**: reputable blogs, talks, and course notes. Fine for leads and intuition; confirm against tier 1–2 before relying on them.
4. **Unreliable**: forums, social media, AI-generated summaries, and uncited claims. Use only as leads, never as evidence.

## Rules
- **Read the source itself before citing it.** Don't cite a paper based only on its abstract or someone else's summary of it.
- **Every citation must be real and checkable.** Each one needs a DOI, arXiv ID, or stable URL. Unverified references never go into a writeup.
- **Quote exactly, paraphrase faithfully.** Mark direct quotes and include page or section numbers when you can.
- **Note the date and version** for anything that changes: software docs, websites, datasets, model versions.
- **Track contradicting evidence.** When sources disagree, record both sides and how you resolved it.

## Where things live
- `literature/references.bib`: one BibTeX file per direction, the single source of citation truth.
- `literature/notes/<firstauthor><year>-<short-title>.md`: one note per paper you read, using the template below.

```markdown
# <Title> (<Authors>, <Year>)
- **Link/DOI:**
- **Tier:** 1 / 2 / 3
- **Read depth:** skimmed / read / studied
- **Claim:** what they show, in one or two sentences
- **Method:**
- **Evidence strength / limitations:**
- **Relevance to us:**
- **Key quotes (with page):**
```

Default citation style: author–year in notes, with BibTeX keys in `<firstauthor><year><firstword>` form (for example `vaswani2017attention`). A writeup uses whatever style its venue requires.

There's no reference manager yet. If one gets adopted (Zotero with Better BibTeX is a good fit), it should export to each direction's `references.bib`.

**ML-specific:** arXiv preprints are common and often the primary source. Cite the published version once one exists, and note which version you read (for example `v2`).
