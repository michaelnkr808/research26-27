# Blind baselines for the feasibility tally (2026-11-09)

Written by an agent for the user, 2026-11-09, before any D+R proxy result exists. Claim labels: verified = checked this session against a file or the fetched HTML; recalled = from memory of earlier work; speculative = my reading.

**Why this exists.** The direction STANDARDS ("Methods and evaluation") require every predictor comparison to include (1) the partner's single-encoder score, (3) the paper's objective-overlap reasoning coded as a label before seeing results, and (4) a random ranking. The 2026-11-08 reviews found none of these was computed for the tally (`reviews/2026-11-08-finished-experiments.md` items 6 and 7; `reviews/2026-11-09-synthesis.md` row I15) (verified). This memo computes them. Numbers come from `src/tally_baselines.py` (run with `python -I`; verified, output matches every number below).

**What "blind" means here, plainly.** The label (part A) is written before any D+R proxy-policy result exists, so it cannot be bent toward our own ground truth. It is not blind to the paper: the Table 10 numbers (D+R = 78.59, the lowest of the seven D+X pairs) were already known to the user and to us, and the paper's prose in §4.4 was written after the authors saw those numbers (verified: the prose cites them). So the label tests whether the paper's explanation, applied mechanically, reproduces the paper's own ordering. It says nothing about whether the explanation is true.

Scope and units: partners L = SigLIP 2, S = SAM 3, G = VGGT-Omega, R = C-RADIOv4, all paired with DINOv3 (D). V = V-JEPA 2.1, U = UniTok, H = HRP appear only in part A (no tally run for them). The paper's order of the four tally partners, from `literature/liu2026what-table10-composition.csv` (Table 10 means; verified): L (+1.83) > S (+0.49) > G (+0.40) > R (-5.41), as change in LIBERO-Plus success over D alone (84.0, DINOv3's LIBERO-Plus total from Table 1, `literature/liu2026what-table1-single-source.csv`; the Table 10 CSV has no D-alone row).

## A. Overlap label

### Rule (stated before labelling; derived only from the paper's reasoning)

The sentences relied on, copied from the fetched arXiv HTML v1 (https://arxiv.org/html/2610.13024v1), section numbers as the paper numbers them. Numbers inside the paper's §4.4 sentences were stripped by my text extraction, so they are replaced with "..."; the "..." in the §3.2 quotes stand for citations and omitted words, not numbers:

- §4.4 (RQ3): "The composition results suggest that overlap in pretraining objectives can limit complementary gains."
- §4.4: "Spatial capabilities similarly overlap: DINOv3 already provides dense features supporting segmentation and depth estimation, while VGGT-Omega uses a DINOv3-initialized visual encoder." (the HTML writes VGGT with the Omega glyph dropped in extraction)
- §4.4: "Adding VGGT-Omega or SAM 3 to DINOv3 produces little change, although both improve some larger combinations."
- §4.4: "A direct teacher–student relationship occurs with C-RADIOv4, which distills DINOv3, SigLIP 2, and SAM 3: combining it with DINOv3 reduces LP success to ..."
- §4.4: "Both UniTok and SigLIP 2 learn language-aligned visual representations."
- §4.4: "Adding SigLIP 2 improves DINOv3 by ... percentage points, but reduces success by ... points once UniTok is included" (the paper's SigLIP 2 overlap is with UniTok, not with DINOv3).
- §4.4: "Meanwhile, adding V-JEPA 2.1 to DINOv3 and UniTok improves performance, consistent with its video-based predictive pretraining providing complementary cues."
- §4.4, Takeaway 3: "When choosing additional sources, consider which task-relevant information they may contribute beyond the existing features, accounting for overlap in pretraining objectives and teacher models."
- §3.2: "DINOv3 ... uses self-distillation to learn semantic and dense spatial features"; "C-RADIOv4 ... distills DINOv3, SigLIP 2, and SAM 3."

The idea in words: a second encoder helps when it brings something the first one lacks, and does not help when it is a copy of, or already contained in, what the first one has. The paper names two kinds of overlap with DINOv3.

Rule, applied to a pair D+B, first clause that fires wins:

1. **Harmful** if B is a distilled student of D (direct teacher–student relationship, §4.4).
2. **Neutral** if B's pretraining capability is one the paper says D already provides, or B is built on a DINOv3-initialised encoder (§4.4: "dense features supporting segmentation and depth estimation"; "DINOv3-initialized visual encoder"). An explicit §4.4 statement that B is complementary to a set containing D overrides this clause; this is what keeps V-JEPA 2.1 (§3.2: learns "spatial and temporal relationships") out of clause 2. The four tally-partner labels (L, S, G, R) do not depend on this override and are unchanged.
3. **Helpful** otherwise: no overlap with D is flagged by the paper, so B adds information beyond D (Takeaway 3). "Helpful" here means "no overlap flagged", not "the paper shows a gain".

This is a three-level scheme. The paper's reasoning gives no basis for ordering within a level, so ties stay ties. Inputs used to apply the rule: only B's identity and the §3.2/§4.4 descriptions above. The Table 10 deltas were not consulted while labelling.

### Labels for every DINOv3 pair in the Table 10 CSV

| Pair | Clause that fires | Label |
|---|---|---|
| D+R (C-RADIOv4) | 1: distills DINOv3 | harmful |
| D+S (SAM 3) | 2: segmentation, which D already provides | neutral |
| D+G (VGGT-Omega) | 2: built on DINOv3-initialised encoder; depth in D | neutral |
| D+L (SigLIP 2) | none (paper's SigLIP 2 overlap is with UniTok) | helpful |
| D+U (UniTok) | none (UniTok overlap is with SigLIP 2, not D) | helpful |
| D+V (V-JEPA 2.1) | none (paper calls it complementary, in the D+U context) | helpful |
| D+H (HRP) | none (interaction-oriented; §4.4 flags no overlap) | helpful |

### Comparison with Table 10 (done after labelling; verified from the CSV)

| Pair | Label | Table 10 mean | Change vs D alone (84.0) | Eval-seed SD |
|---|---|---|---|---|
| D+U | helpful | 86.12 | +2.12 | 0.32 |
| D+L | helpful | 85.83 | +1.83 | 0.28 |
| D+V | helpful | 84.93 | +0.93 | 0.28 |
| D+S | neutral | 84.49 | +0.49 | 0.61 |
| D+H | helpful | 84.48 | +0.48 | 0.74 |
| D+G | neutral | 84.40 | +0.40 | 0.13 |
| D+R | harmful | 78.59 | -5.41 | 1.01 |

Reading it: the label gets the sign direction of the one harmful pair and puts all four "helpful" pairs above all "neutral" pairs except D+H, which sits at the same size as D+S (+0.48 vs +0.49) and well inside the evaluation-seed SD (0.74). The SD is across three evaluation seeds of one trained checkpoint, not across training runs (recalled from the repo note, verified there against App. C.2 and E.3), so it understates real run-to-run noise. On the four tally partners, labels L=helpful, S=neutral, G=neutral, R=harmful give a Spearman rho of +0.95 against the paper order (ties use average ranks, so this is not directly comparable to the strict-ordering distribution in part B; it is above all 24 strict orderings except the perfect one, speculative as a placement).

Limit specific to this label: I chose to call "no overlap flagged" helpful. A stricter reader could call L, U, H "neutral" (concrete basis: §3.2 describes SigLIP 2 as learning "semantic and dense visual features", almost the same words as DINOv3's "semantic and dense spatial features") and only V helpful. Under that alternative the four-partner label becomes L=S=G=neutral, R=harmful, which still puts R last but cannot order L, S, G (speculative; not computed).

## B. Random-ranking baseline (n = 4 partners)

With 4 partners there are 4! = 24 orderings. Spearman rho for n = 4 is 1 - 6*sum(d^2)/(4*15) = 1 - sum(d^2)/10, against the paper's order L > S > G > R. Distribution over the 24 (verified, script output):

| rho | orderings (of 24) | P(rho >= this) |
|---|---|---|
| +1.0 | 1 | 1/24 |
| +0.8 | 3 | 4/24 |
| +0.6 | 1 | 5/24 |
| +0.4 | 4 | 9/24 |
| +0.2 | 2 | 11/24 |
| 0.0 | 2 | 13/24 |
| -0.2 | 2 | 15/24 |
| -0.4 | 4 | 19/24 |
| -0.6 | 1 | 20/24 |
| -0.8 | 3 | 23/24 |
| -1.0 | 1 | 24/24 |

Mean rho is 0. Chance a random ordering puts R last: fix R in position 4, order the other three freely, 3! = 6 orderings out of 4! = 24, so 6/24 = 1/4 (script confirms 6/24). Any single predictor getting R last is therefore not surprising: one time in four from pure chance.

## C. Solo-score and other predictors on the same footing

Higher = predicted better partner, except CKA, where lower CKA(D, B) is read as the better partner (the "different = complementary" reading, which mentor point 1 in STANDARDS warns against treating as established). Values are from the four JSONs (`experiments/2026-10-19-feasibility-tally/results/tally-libero_goal-dinov3-*.json`), verified by the script.

**Global features (primary)**

| Predictor | L | S | G | R | Order, best to worst | rho vs paper | R's rank | P(random rho <= obs) / P(>= obs) |
|---|---|---|---|---|---|---|---|---|
| solo R2(B) | 0.609 | 0.654 | 0.617 | 0.670 | R > S > G > L | -0.8 | 1 | 4/24 / 23/24 |
| gain(B given D) | 0.017 | 0.033 | 0.014 | 0.037 | R > S > L > G | -0.4 | 1 | 9/24 / 19/24 |
| gain(D given B) | 0.057 | 0.029 | 0.047 | 0.017 | L > G > S > R | +0.8 | 4 | 23/24 / 4/24 |
| CKA, agentview (lower better) | 0.576 | 0.559 | 0.213 | 0.583 | G > S > L > R | +0.2 | 4 | 15/24 / 11/24 |
| CKA, wrist (lower better) | 0.792 | 0.643 | 0.267 | 0.854 | G > S > L > R | +0.2 | 4 | 15/24 / 11/24 |
| CKA, mean of both cams | 0.684 | 0.601 | 0.240 | 0.719 | G > S > L > R | +0.2 | 4 | 15/24 / 11/24 |

**Mean features (secondary)**

| Predictor | L | S | G | R | Order, best to worst | rho vs paper | R's rank | P(random rho <= obs) / P(>= obs) |
|---|---|---|---|---|---|---|---|---|
| solo R2(B) | 0.645 | 0.654 | 0.627 | 0.668 | R > S > L > G | -0.4 | 1 | 9/24 / 19/24 |
| gain(B given D) | 0.034 | 0.035 | 0.016 | 0.037 | R > S > L > G | -0.4 | 1 | 9/24 / 19/24 |
| gain(D given B) | 0.031 | 0.024 | 0.031 | 0.011 | G > L > S > R | +0.4 | 4 | 19/24 / 9/24 |
| CKA, agentview (lower better) | 0.672 | 0.665 | 0.268 | 0.766 | G > S > L > R | +0.2 | 4 | 15/24 / 11/24 |
| CKA, wrist (lower better) | 0.782 | 0.665 | 0.399 | 0.863 | G > S > L > R | +0.2 | 4 | 15/24 / 11/24 |
| CKA, mean of both cams | 0.727 | 0.665 | 0.334 | 0.814 | G > S > L > R | +0.2 | 4 | 15/24 / 11/24 |

Notes, each verified from the script output unless marked:

- **gain(D given B) restates B's solo score.** By definition gain(D|B) = R2(D+B) - R2(B) (the script asserts this for all eight cases to within 1e-9). R2(D+B) varies little across partners (global 0.664 to 0.687), so gain(D|B) is mostly the solo score upside-down. Here it is exactly the reverse of the solo order in both feature types (global: solo R > S > G > L, gain(D|B) L > G > S > R; mean: solo R > S > L > G, gain(D|B) G > L > S > R, where L and G differ by 0.0001, a tie in practice). So gain(D|B) putting R last is the same fact as R having the highest solo score. This is the FE-review point (`reviews/2026-11-08-finished-experiments.md`, item 7).
- **The solo score ranks R first, the paper ranks R last.** The most plausible reading of rho = -0.8 (global) is that, in this setup, a partner that is strong alone is not a better partner for D. The paper's order is close to the reverse of the solo order (speculative as to cause; descriptive as to fact).
- **Where each rho sits.** Under chance, rho >= +0.8 happens in 4/24 orderings and rho <= -0.8 in 4/24. gain(D|B) global (+0.8) and solo score global (-0.8) are each at the 4/24 edge of the random distribution. Everything else is in the middle of it (CKA at +0.2 is matched or beaten by 11/24 random orderings, so unremarkable).
- **CKA** gets R last in every cell, but this rests on R's CKA being highest by a small margin in the global agentview cell (0.583 vs L 0.576; no CIs were computed for CKA, per the tally README). Its rho of +0.2 comes from ranking G first (G has by far the lowest CKA and a small paper gain) and S above L (paper has L above S).
- **The overlap label** reaches rho +0.95 on the same four partners (part A), above every predictor in the tables. It is the only predictor that does not use our features at all.

## D. What this means (plain words)

The paper explains the one harmful pair with a story: C-RADIOv4 was trained by copying DINOv3 (among others), so adding it gives DINOv3 nothing new. Written down as a rule before any proxy result, that story reproduces the paper's ordering of the four partners almost exactly. That tells us the story is consistent with the paper's own numbers, not that it is right.

Our frozen-feature measures do not do as well. The simplest baseline, how good each partner is on its own, points the wrong way: C-RADIOv4 is the best single encoder in our probe and the paper's worst partner. The one measure that puts C-RADIOv4 last, gain(D|B), does so only because it is that same solo score upside-down. CKA also puts it last, but by a margin the tally cannot tell from a tie. And getting C-RADIOv4 last is something a random shuffle does one time in four. So none of the tally's measures has shown it carries more information than a coin flip about which partner will help. For the project this means any later claim that a feature measure predicts proxy-policy gains has to beat these numbers, and the objective-overlap label is the bar to beat, not zero.

## E. Limits

- n = 4 partners (7 in the label table). Descriptive only; no significance is claimed, as STANDARDS says ("With 7 pairs, report the tally and don't claim significance"). One ordering swap changes rho by 0.2 to 0.4.
- Ground truth is one trained checkpoint per Table 10 row, with SDs over three evaluation seeds (not training seeds), on perturbed LIBERO-Plus. Our probes use clean libero_goal frames and single-step actions (recalled from the tally README caveats). The paper's order is therefore itself noisy, particularly among S, G (+0.49 vs +0.40, inside the SDs).
- The label rule was fixed knowing the paper's results and its prose (above). A human reader could reasonably draw the lines differently (see the alternative in part A).
- Probe dimensions differ across pairs and no matched-dimension control was run (tally README caveat), so solo and gain values carry that confound too.
- Random baseline here is uniform over strict orderings; predictors with ties (the label) are placed on it only approximately.
- Not run: V, U, H tally predictors (not extracted), a control partner (noise or duplicate), and Tsagkas et al.'s whole-frame statistic (the synthesis adds it for the later comparison).
