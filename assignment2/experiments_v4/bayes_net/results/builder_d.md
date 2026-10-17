# Builder D — DAG structure + CPT smoothing / calibration (Small Bayesian Network)

**Scope.** Features FIXED = the incumbent's 12 features
(8 grid densities + `euler_topology` + `vertical_runs` + `aspect_ratio` +
`vertical_dominance`). I changed only the DAG over those features and the CPT
estimator. `FEATURE_NAMES`, `FEATURE_CARDINALITIES` and `extract_features` are
byte-for-byte identical to `assignment2/bayes_net.py` (asserted in
`builder_d_experiments.verify_features`).

**Protocol.** `v4_common.load_split()` → 49,999 train / 10,001 validation,
seed 1322; only the 60,000 training images are ever read. Test set untouched.

**Reproduction of the incumbent.** `bayes_net.BayesNetClassifier(alpha=1.0)` →
val accuracy **0.8614**, exactly `BASELINE.md`. My cached-feature `DagModel`
with the incumbent's `FEATURE_PARENTS` also gives 0.8614, bit-identical
predictions (stage 0 regression).

## Verdict: **ADOPT** (with an honest caveat on the effect size)

| | incumbent (v3) | candidate (`builder_d_serpentine`) |
|:---|---:|---:|
| val acc, seed 1322 | 0.8614 | **0.8653** (+0.39pt) |
| val acc, seed 7 | 0.8455 | **0.8497** (+0.42pt) |
| val acc, seed 2024 | 0.8586 | **0.8615** (+0.29pt) |
| mean over 3 seeds | 0.8552 | **0.8588** (+0.37pt) |
| macro F1, seed 1322 | 0.8589 | **0.8630** |
| McNemar, seed 1322 | — | b=205 c=166, **p=0.0484** |
| McNemar, seed 7 | — | b=235 c=193, **p=0.0474** |
| McNemar, seed 2024 | — | b=210 c=181, p=0.1567 |
| **pooled McNemar (3 seeds)** | — | b=650 c=540, **p=0.0016**, χ²=9.98 |
| bootstrap 95% CI (seed 1322) | — | **[+0.020pt, +0.770pt]** |

The gain is positive on all three independent splits, significant on two of
three individually and strongly significant pooled. McNemar on seed 1322 alone
is p=0.048, i.e. only *just* below 0.05 — this is a small, real-but-modest
effect, not a rout. It is a **structural** win (see below), not a
smoothing-hyperparameter win.

## Best DAG: "contour serpentine + orientation chain" (10 feature edges)

```
FEATURE_PARENTS = (-1, 0, 1, 6, 3, 2, 7, 5, -1, 11, 9, 8)

tl->tc->tr->mr->br->bl->ml->mc          (grid contour serpentine)
euler -> vertical_dominance -> vertical_runs -> aspect_ratio   (orientation chain)
```

(Indices: 0..7 = tl,tc,tr,ml,mc,mr,bl,br; 8=euler, 9=vruns, 10=aspect, 11=vdom.)

**Edge-by-edge semantic justification.**

Grid contour serpentine (clockwise from the top-left cell):
- `tl→tc→tr` — left-to-right across the top band, same as the incumbent row
  chain; adjacent top cells of a digit share the same stroke (e.g. the top of a
  `0/2/3/5/6/8/9`), so this is a genuine local continuity edge.
- `tr→mr` — continues down the **right side** of the glyph. On MNIST the right
  edge is where the loop of `0/6/8/9` closes and where `5/3` terminate.
- `mr→br` — continues down to the bottom-right corner (kept from the incumbent).
- `br→bl` — right-to-left across the **bottom band** (kept from the incumbent).
- `bl→ml` — **NEW (the key change)**: closes the left side by running *up* the
  left column, mirroring the already-successful `mr→br` link. The incumbent's
  `ml` was a chain root; giving it the bottom-left cell as a parent makes the
  left column a continuity edge instead of an orphan start.
- `ml→mc` — enters the interior last, from the left. The centre cell is
  explained by its left neighbour rather than starting an independent run.

Net effect vs. the incumbent: instead of three disconnected left→right strips
(`tl→tc→tr`, `ml→mc→mr`, `br→bl` + `br→mr`), the 8 grid cells now form **one
contour chain** that traverses the digit's boundary and only then goes inward.
The added `bl→ml` edge is what buys the accuracy; see the ablation table.

Semantic orientation chain:
- `euler→vdom` — how many strokes the glyph has (topology: components − holes)
  determines how vertically dominated the ink gradient is. A single-stroke
  glyph (`1`) is gradient-vertical; a looped glyph (`0/8`) mixes directions.
- `vdom→vruns` — vertical-gradient dominance determines how many vertical runs
  survive per occupied column (fewer near-horizontal crossings).
- `vruns→aspect` — the run structure plus the extent fixes the observed aspect
  ratio.

This ordering (`euler → vdom → vruns → aspect`) beat the incumbent ordering
(`euler → vruns → {aspect, vdom}`) by +0.19pt on the same grid (0.8653 vs
0.8634, seed 1322) — a single extra "arrow flip", so within-noise on one seed,
but it was the best of 7 hand-wired semantic orders and is at least as
defensible causally.

## Full variant tables (seed 1322, `scheme=marginal`, `alpha=1.0`)

### A. Grid-only DAG variants (incumbent semantic chain held fixed)

| grid variant | val acc | edges | note |
|:---|---:|---:|:---|
| **G_serp** | **0.8634** | 10 | single contour serpentine (winner) |
| G_ml_bl | 0.8627 | 10 | incumbent rows + `bl→ml` only |
| G_ml_br | 0.8627 | 10 | `br→ml` cross-row link |
| G_diag_tl_mc | 0.8623 | 10 | `tl→mc` diagonal leak |
| G_ml_tl | 0.8616 | 10 | `tl→ml` closes left column downward |
| G_incumbent | 0.8614 | 9 | v3 baseline |
| G_raster | 0.8582 | 10 | raster reading order |
| G_boustro_rev | 0.8552 | 10 | boustrophedon, bottom R→L |
| G_hub_mc | 0.8547 | 10 | middle-centre as hub |
| G_boustro | 0.8536 | 10 | boustrophedon, bottom L→R |

Reading: closing the **left** column via the **bottom** cell (`bl→ml`, or the
full serpentine) helps; closing it via the **top** cell (`tl→ml`) does not.
Raster / boustrophedon scans that cross the image discontinuously are clearly
worse — the useful edges are *spatially local* (adjacent cells / shared edges),
which is the expected result and is worth stating in the report.

### B. Semantic wiring variants (on the winning G_serp grid)

| semantic variant | chain | val acc |
|:---|:---|---:|
| **S_evra** | euler→vdom→vruns→aspect | **0.8653** |
| S_eavr | euler→aspect→vruns→vdom | 0.8643 |
| S_vr_vdom | vdom→vruns→aspect | 0.8639 |
| S_vr_hub | vruns hub, euler child of vruns | 0.8635 |
| S_inc | euler→vruns→{aspect,vdom} (incumbent) | 0.8634 |
| S_serial | euler→vruns→aspect→vdom | 0.8618 |
| S_v2 | two independent edges (v2 style) | 0.8610 |

70 grid×semantic combinations were scored; `G_serp|S_evra` was best.

### C. Smoothing / calibration variants (on the final DAG)

Alpha is irrelevant here too, as v3 found. All sensible settings are within
±0.05pt (≈5 validation examples) — noise. The **DAG**, not the smoother, is the
effect.

| variant | val acc |
|:---|---:|
| `A_marginal` alpha=1.0 (incumbent scheme) | 0.8653 |
| `B_uniform` alpha=1 (no backoff) | 0.8650 |
| `B_grand` alpha=1 (back off to grand-mean state profile) | 0.8656 |
| `B_parent_marginal` alpha=1 (back off to P(F\|Pa)) | 0.8658 |
| `D_uniform class prior` | 0.8640 |
| `D_prior` alpha ∈ {0.01, 0.1, 10} | 0.8653 |
| `E_EB` per-node alpha (type-II ML, internal split) | 0.8654 |
| `E_EB` pooled alpha | 0.8653 |
| `F_pow` renormalized powers, log-loss-fitted | 0.8611 |
| `F_pow` renormalized powers, accuracy-fitted | 0.8625 |
| `G_temper` raw log-likelihood powers **[EXT]** | 0.8654 |

Alpha sweeps on the final DAG: 0.1/0.3/0.5/1.0/2.0/5.0 → 0.8658/0.8657/0.8655/
0.8653/0.8654/0.8652 (monotone in 1e-4, i.e. flat).

Cross-seed check of the smoother at different settings (the key falsification):

| config | seed 1322 | seed 7 | seed 2024 |
|:---|---:|---:|---:|
| marginal a=1.0 | 0.8653 | 0.8497 | 0.8615 |
| marginal a=0.1 | 0.8658 | 0.8495 | 0.8618 |
| parent_marginal a=1.0 | 0.8658 | 0.8496 | 0.8614 |
| uniform a=1.0 | 0.8650 | 0.8493 | 0.8613 |

No configuration dominates across all three seeds → no smoother is
demonstrably better than the incumbent's class-conditional-marginal backoff.
I therefore **keep the incumbent's smoother unchanged** (alpha=1.0, backoff to
the class-conditional marginal) and change only the DAG. That keeps the
adoption minimal and fully explainable, per PDF §5.

### D. Smoothing details tested and rejected

- **Per-node empirical Bayes (type-II ML) alpha.** The DAG joint
  log-likelihood factorizes per node given the observed `(y, parent)` columns,
  so each node's alpha can be picked independently by held-out likelihood on a
  train-internal 75/25 split. Selected alphas were highly non-uniform
  (0.01 … 30), i.e. the procedure *does* detect per-node differences — but the
  resulting model is **not** better (0.8654 vs 0.8653), and the selected values
  swing wildly between seeds (0.01 at one node, 30 at another). Interpretation:
  with 50k training rows the cell counts are large enough that alpha is a
  vanishing perturbation inside the model's own likelihood, so type-II ML has
  almost no signal to fit and mostly chases sampling noise. **Rejected** —
  added complexity, no gain, unstable.
- **Backoff to `P(F|Pa)` / grand mean instead of `P(F|y)`.** +0.05pt on seed
  1322 (0.8658), −0.01pt on seed 2024. **Rejected**: not reproducible, and
  backing off to a *class-independent* target loses exactly the label
  information the CPT is supposed to carry; the incumbent target is the more
  principled one.
- **Jelinek-Mercer interpolation.** Best at λ=0.97 (0.8658) and degrades fast
  below λ=0.9; at λ→1 it *is* the incumbent scheme, so this is just
  rediscovering "almost no smoothing". **Rejected**: no distinct mechanism, no
  gain.
- **Uniform class prior.** 0.8640, clearly worse than the (near-uniform anyway)
  empirical prior. **Rejected**; class prior smoothing alpha ∈ {0.01,0.1,10} is
  flat at 0.8653, so prior smoothing also does not matter.
- **Renormalized per-feature powers `P(F|Pa)^{w_i}`.** Fitted coordinate-wise
  on a train-internal split by held-out log-likelihood; optimal w_i ≈ 0.7–0.85
  (shrink every feature — the classic double-counting fingerprint of correlated
  features). This is a **valid CPT** (renormalized, so prediction keeps the
  exact `log P(y)+Σ log P(F|Pa)` form) yet it made accuracy **worse** (0.8611
  log-loss-fitted, 0.8625 accuracy-fitted). Interpretation: shrinking all
  likelihoods toward uniform mainly *flattens the decision margins*; it improves
  held-out log-loss but costs hard-decision accuracy. **Rejected.**
- **Raw tempered log-likelihood `Σ w_i log P(F|Pa)` [declared EXTENSION].**
  This leaves the joint-probability reading (factors are no longer normalized
  CPTs), so it is out of the pure DAG/CPT family. It scored 0.8654 — again no
  gain — so there is no reason to adopt it and I do not. Reported only for
  completeness.

## Confusion structure (seed 1322)

Incumbent top pairs: 5→3 (104), 3→5 (74), 3→2 (73), 5→2 (69), 2→3 (65).
Candidate top pairs: 5→3 (122), 3→2 (77), 3→5 (75), 2→3 (62), 4→9 (59), 5→2 (56).

The candidate **improves 5→2 (69→56), 2→3 (65→62), 4→9**, but **worsens 5→3
(104→122)**. Net correct +39 examples. The contour chain helps the
upper-left→lower-right shape confusions but slightly strengthens the 5/3 pair;
worth one sentence in the report as a known trade-off.

## Files

- `candidates/builder_d_serpentine.py` — the ADOPT candidate; drop-in for
  `bayes_net.py` (same public API, `FEATURE_PARENTS = (-1,0,1,6,3,2,7,5,-1,11,9,8)`).
- `candidates/builder_d_experiments.py` — reproduces every table here:
  `python builder_d_experiments.py --stages grid,cross,refine,smoothing,crossseed,regress`.
- `candidates/builder_d_final_validation.py` — end-to-end head-to-head with the
  real modules (McNemar, bootstrap, cross-seed, bounds audit):
  `python builder_d_final_validation.py`.
- `results/builder_d_stage_raw_seed1322.json`, `results/builder_d_final.json` —
  raw numbers.

## Compliance audit

`check_bn_bounds(FEATURE_NAMES, FEATURE_CARDINALITIES, FEATURE_PARENTS)` → OK:
12 features, each cardinality 4–5 (in 2–5), 10 feature-feature edges (≥2), Y is
a parent of every feature and each feature has ≤1 feature parent (≤2 total),
the feature graph is a forest of two directed paths hence acyclic, no hidden
variables, no automatic structure learning (all 70+ candidates were hand-wired
from a small semantic set), every CPT smoothed, prediction is
`argmax_y [log P(y) + Σ_i log P(F_i | Parents(F_i))]`.
