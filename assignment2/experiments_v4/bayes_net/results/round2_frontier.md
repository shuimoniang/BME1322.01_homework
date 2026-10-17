# Round 2 — BN frontier push (final)

**Task.** Final adversarial push on the Small Bayesian Network, strictly inside the
assignment PDF §4.2 rules. Two genuinely unexplored levers were mandated:

- **(A) conditional / position-relative features** — features defined *conditionally
  on the image's own ink content* (bounding-box-normalised), not absolute windows;
- **(B) marginal structural refinements on the confirmed champion M6**, adjudicated
  with an 8-seed pooled McNemar **Bonferroni-corrected over the number of variants tried**;
- plus a quick check of the class prior (empirical vs uniform) and the
  class-conditional-marginal backoff vs plain uniform Lidstone under M6.

**Protocol.** `v4_common.load_split()` → 49,999 train / 10,001 validation.
Search split = seed 1322. Independent replication = seeds 7, 2024, 1, 42, 99, 123, 555.
The pooled McNemar that adjudicates a gain uses the **7 held-out seeds only** (the search
split must not be re-counted as evidence). Only the 60,000 training images are ever read;
`t10k-*.gz` is never opened. Discretisation thresholds are fit on training data only.
All structures are named members of small hand-authored families — **no automatic
structure learning** (no Hill-Climbing / PC / Chow-Liu / score search).

**Headline.** Idea (A) is a **real, large, fully replicated effect** and produces a
**NEW CHAMPION**. Idea (B) produces nothing that clears the corrected gate — M6's DAG is
already at the frontier. The champion's estimator (class prior + backoff) is unchanged.

**Reproduce:** `python round2_frontier_experiments.py` (add `--quick` for seed 1322).
Raw numbers: `cache/round2_frontier_raw.json`, console log `cache/r2_frontier_run.log`.

---

## 0. Baseline reproduction (all 8 seeds)

| seed | incumbent v3 | M6 |
|---:|---:|---:|
| 1322 | 0.8614 | 0.8686 |
| 7 | 0.8455 | 0.8553 |
| 2024 | 0.8586 | 0.8662 |
| 1 | 0.8561 | 0.8641 |
| 42 | 0.8538 | 0.8609 |
| 99 | 0.8580 | 0.8678 |
| 123 | 0.8559 | 0.8595 |
| 555 | 0.8561 | 0.8616 |

Bit-identical to `BASELINE.md`, `builder_c.md` and `critic_beta.md`. The M6 champion is
reproduced exactly (seed 1322 = 0.8686), and my cached sweep engine agrees with the real
`builder_c_dag_m6.BayesNetClassifier` module to the digit on all 8 seeds.

---

## Idea (A) — content-relative (bounding-box-normalised) features

### A.1 The modelling idea (what is genuinely new here)

The incumbent's (and M6's) eight cell densities are **absolute** region means: fixed
rows/cols 7..22 of the 28×28 frame. A `7` written small and centred, and a `7` written
large and shifted, deposit their ink in *different* fixed windows, so an absolute cell mean
mixes **shape** with **where and how large the digit was written**. The new features measure
the *same* quantity in the image's **own ink bounding box**, removing position and scale.

This is a re-definition of the *variable*, not a tuning knob: each is one discrete feature
with a one-sentence semantic and 2–5 states (here 5). Builder C tested a fixed-window
geometry sweep and a "true-bbox aspect", but **never** tested content-relative *density*
features — this is genuinely new territory (confirmed by reading `_sweep1_single.py`,
`_sweep4_refine.py`, `builder_c.md` §4).

### A.2 Semantic definition of the new feature family (all one continuous statistic each)

Let the ink bbox be rows `[top, bottom]`, cols `[left, right]`, height `H`, width `W`.
For a pixel at offset `(dr, dc)` inside the box, its normalised band is
`(floor(3·dr/H), floor(3·dc/W)) ∈ {0,1,2}²`.

| key | semantic (one sentence) | definition |
|---|---|---|
| `bbXY` | ink density of the (X∈{T,M,B}, Y∈{L,C,R}) cell of the glyph's **normalised 3×3 bbox grid** | `count(ink in band) / (band height · band width)` |
| `bbQXY` | same for the four normalised 2×2 **bbox quadrants** | as above |
| `bbRowband`/`bbColband` | ink density of the top/middle/bottom (resp. left/middle/right) third of the bbox | as above |
| `bbCR`, `bbCC` | vertical / horizontal centroid of the ink **inside the bbox**, normalised 0→1 | `(centroid − top)/(H−1)` etc. |
| `bbPR`, `bbPC` | normalised position of the **fullest ink row / column** inside the bbox | `argmax row-profile` mapped to 0→1 |

Discretisation: **`zero_quantile` at K=5** for `bbXY`/`bbQ*/bb*band` (state 0 = exactly zero
ink, states 1..4 = training quantiles of the positives) — the incumbent's own rule; plain
`quantile` K=5 for the four position statistics. Cardinality 5 (in 2..5). Thresholds
estimated on training data only. Cached to `cache/round2_content.npz` (60,000 images, 1.4 s).

**The winning change replaces exactly TWO grid cells** (`density_middle_right`,
`density_bottom_right`) by their bbox-normalised versions — 12 features total, all card 4–5.

### A.3 Why these two cells (evidence, train-only)

Marginal informativeness `I(F;Y)` (K=5, train split, seed 1322) and conditional redundancy
with the other six retained absolute cells:

| cell | `I(abs;Y)` | `I(bbox;Y)` | `Σ I(abs; other_abs │ Y)` | `Σ I(bbox; other_abs │ Y)` |
|---|---:|---:|---:|---:|
| mid-right (`mr`) | 0.2405 | 0.1871 | **0.4060** | **0.2755** |
| bot-right (`br`) | 0.2950 | 0.2152 | **0.3338** | **0.3007** |

The bbox frame makes the middle/bottom-right cells **decorrelated from the retained
absolute cells** (a 33 % / 10 % drop in class-conditional redundancy) while keeping
comparable marginal information — i.e. it converts a redundant *position* signal into
complementary *shape* signal. This is the mechanism, and it is exactly the sort of
"Why?" the PDF §5 asks the report to explain.

### A.4 Main table — replacing k of the 8 grid cells by their bbox-normalised versions

Every row is a complete, legal 12-feature design with the M6 DAG unchanged.
`p_one` = exact one-sided pooled McNemar vs M6 over the **7 held-out seeds** (b = M6 wrong &
variant right); `p_bonf` = `p_one × m` with `m` = the family size actually tried.

| design | replaces | 8-seed mean | min | Δ mean vs M6 | beats | pooled b/c | p_one | Bonferroni m | p_bonf | verdict |
|---|---|---:|---:|---:|:--:|---|---:|---:|---:|:--|
| M6 (reference) | — | 0.8630 | 0.8553 | — | — | — | — | — | — | incumbent champ |
| bbox_br | 1 cell | 0.8682 | 0.8605 | +0.0051 | 8/8 | 2005/1626 | 1.7e-10 | 8 | 1.4e-9 | **✓** |
| bbox_mr | 1 cell | 0.8677 | 0.8610 | +0.0046 | 8/8 | 1726/1400 | 3.0e-09 | 8 | 2.4e-8 | ✓ |
| bbox_tr | 1 cell | 0.8671 | 0.8597 | +0.0040 | 8/8 | 2021/1755 | 8.0e-06 | 8 | 6.4e-5 | ✓ |
| **bbox_mr+br** | **2 cells** | **0.8778** | **0.8704** | **+0.0147** | **8/8** | **2876/1830** | **4.0e-53** | **28** | **1.1e-51** | **✓✓ (CHAMPION)** |
| bbox_ml+br | 2 cells | 0.8701 | 0.8611 | +0.0070 | 8/8 | 2796/2283 | 3.2e-13 | 28 | 9.1e-12 | ✓ |
| bbox_tr+mr | 2 cells | 0.8699 | 0.8652 | +0.0069 | 8/8 | 2696/2206 | 1.4e-12 | 28 | 3.9e-11 | ✓ |
| bbox_ml+mr+br | 3 cells | 0.8785 | 0.8727 | +0.0155 | 8/8 | 3510/2411 | 9.4e-47 | 56 | 5.3e-45 | ✓ |
| bbox_tr+mr+bl+br | 4 cells | 0.8756 | 0.8697 | +0.0126 | 8/8 | 4041/3122 | 8.8e-28 | 70 | 6.2e-26 | ✓ |
| bbox_tr+ml+mr+bl+br | 5 cells | 0.8803 | 0.8761 | +0.0173 | 8/8 | 4467/3242 | 1.2e-44 | 56 | 6.7e-43 | ✓ |
| bbox_all (all 8) | 8 cells | 0.8662 | 0.8614 | +0.0032 | 8/8 | 4666/4412 | 4.0e-03 | **92** | **0.364** | ✗ not significant |
| all 92 (whole A family) — best member | 5 cells | 0.8803 | 0.8761 | +0.0173 | 8/8 | 4467/3242 | 1.2e-44 | 92 | 1.1e-42 | ✓ |

Reading:

1. **`bbox_mr+br` (the two-feature design) is the champion.** +1.47 pt on the 8-seed mean,
   positive on 8/8 seeds, min margin +1.31 pt, pooled held-out McNemar p = 4.0e-53, and it
   survives Bonferroni for the full **m = 92** family (p_bonf = 3.7e-51). This clears the
   mandate's corrected gate by ~48 orders of magnitude.
2. **A non-monotone optimum.** Going all the way to k = 8 is *not* best: the two weakest
   absolute cells (`tl`, `tc`) genuinely benefit from absolute position, so replacing them
   costs more than it gains. The `k = 8` uniform variant is only +0.32 pt and **fails** the
   corrected gate (p_bonf = 0.364) — a clean illustration that "more of the same idea"
   is not automatically better, and that the effect is a *specific* two-cell re-definition.
3. **k = 5 is also significant** (0.8803, +1.73 pt) but replaces 5 of 8 cells, exceeds the
   mandated 1–2 feature scope, and buys only +0.25 pt over k = 2 (`p = 1.0e-2`). Per PDF §5
   ("don't ship +0.1 pt that makes the report harder to explain"), the two-cell design is
   the right adoption. k = 5 is reported as a documented extension only.

### A.5 Control: is it position, scale, or both? (This is the load-bearing ablation.)

Same 3×3 grid, three coordinate frames. `TRANSLATE-only` = the fixed 15×15 window merely
**shifted** so the bbox centre lands on the window centre (position normalised, scale not);
`SCALE-only` = a window at the image centre whose cells are **sized** by the bbox extent
(scale normalised, position not).

| design | absolute | bbox (pos+scale) | translate-only | scale-only |
|---|---:|---:|---:|---:|
| k=2 (`mr+br`) | 0.8630 | **0.8778** | 0.8712 | 0.8642 |
| k=5 | 0.8630 | **0.8803** | 0.8685 | 0.8501 |

The gain decomposes as **position normalisation ≈ 55–60 % of it, scale normalisation the
rest**; neither alone recovers the full effect, and scale-only applied broadly can hurt.
So the bbox frame — not merely "recentring" — is doing the work. This rules out the
trivial explanation that we just moved the digit to a canonical location.

### A.6 Robustness of the k=2 champion

| check | result |
|---|---|
| binarisation threshold ∈ {48, 64, 80, 96, 128} | 0.8807 … 0.8827 on seed 1322; 0.8704 … 0.8731 on seed 7 — gain present at every threshold, never depends on 64 |
| discretiser: `zero_quantile` K=5 (chosen) | 0.8778 mean |
| discretiser: `quantile` K=5 | 0.8680 — the zero-separation rule matters (to be expected: many bbox cells are exactly 0) |
| cardinality K=4 | 0.8762 — still well above M6; K=5 chosen |
| M6 DAG still best on the new features | M6 0.8778 > serpentine 0.8759 > M6+`bl←ml` 0.8766 > incumbent DAG 0.8738 — the M6 structure is **not** a relic; it still adds ~0.4–0.5 pt on top of the new features |
| Idea-B refinements on top of k=2 (vdom 5 states) | 0.8778 — no change (see B) |

---

## Idea (B) — marginal structural refinements on M6

### B.1 What is testable (named single-edit family)

A pure **edge addition** is structurally impossible: only `euler` has a spare parent slot
(≤2 parents incl. Y), and every legal arc into `euler` closes a cycle through the existing
chains (brute force over all 12·12·12 parent triples → 0 acyclic DAGs, confirmed by the
critic in `critic_beta.md` §2b). So every named edit is a **single-edge re-point** or a
cardinality change. Eight hand-authored members:

| variant | edit | semantic |
|---|---|---|
| B1_aspect5 | `aspect_ratio` 4→5 states | finer width/height resolution |
| B2_vdom5 | `vertical_dominance` 4→5 states | the critic's +0.1 pt lead, re-tested on 8 seeds |
| B3_both5 | both 4→5 states | combined |
| B4_mc_vruns | `mc ← vertical_runs` (was `← ml`) | centre explained by stroke fragmentation, not the left cell |
| B5_mr_vruns | `mr ← vertical_runs` (was `← euler`) | mid-right from fragmentation rather than topology |
| B6_tc_euler | `tc ← euler_topology` (was `← tl`) | top-centre from global topology |
| B7_bl_euler | `bl ← euler_topology` (was `← br`) | bottom-left from global topology |
| B8_orientation_chain | `aspect ← vdom`, `vdom ← euler` | the "orientation chain" re-order (Builder D's `S_evra`) |

### B.2 Result — nothing clears the gate

8-seed means vs M6; `p_one` = pooled exact one-sided McNemar over the 7 held-out seeds;
`p_bonf = p_one × 8` (Bonferroni over the 8 variants actually tried).

| variant | 8-seed mean | min | Δ mean | beats | pooled b/c | p_one | p_bonf | verdict |
|---|---:|---:|---:|:--:|---|---:|---:|:--|
| B2_vdom5 | 0.8637 | 0.8555 | +0.0007 | 7/8 | 817/768 | 0.114 | 0.912 | ✗ |
| B3_both5 | 0.8637 | 0.8555 | +0.0007 | 7/8 | 817/768 | 0.114 | 0.912 | ✗ |
| B5_mr_vruns | 0.8636 | 0.8555 | +0.0006 | 6/8 | 871/806 | 0.059 | 0.472 | ✗ |
| B1_aspect5 | 0.8630 | 0.8553 | +0.0000 | 0/8 | 0/0 | 1.000 | 1.000 | ✗ (prediction-identical) |
| B4_mc_vruns | 0.8624 | 0.8543 | −0.0006 | 3/8 | 778/800 | 0.299 | 1.000 | ✗ |
| B7_bl_euler | 0.8620 | 0.8559 | −0.0010 | 2/8 | 577/641 | 0.036 | 0.284 | ✗ |
| B8_orientation_chain | 0.8617 | 0.8532 | −0.0013 | 0/8 | 601/691 | 0.007 | 0.053 | ✗ |
| B6_tc_euler | 0.8590 | 0.8532 | −0.0040 | 0/8 | 990/1278 | 8e-10 | 0.000 | ✗ |

**Verdict: none clears the corrected gate.** The best variant (`vdom` 5 states, the critic's
suspected +0.1 pt) is +0.07 pt with p_bonf = 0.91 — noise. Note `B1_aspect5` is
prediction-identical to M6: the aspect distribution has an empty 4th bin, so K=5 adds an
unused state (this is why it was flat at 0.8630 with b = c = 0). Every named structural
re-point is neutral or negative. **M6's DAG is saturated**; confirmed on 8 seeds, not 3.

### B.3 Estimator check under M6 (class prior × CPT scheme)

Under the unchanged M6 DAG/features, `p_one` = pooled held-out McNemar vs the incumbent
scheme (empirical prior + class-conditional-marginal backoff, α=1.0):

| scheme | 8-seed mean | min | beats | Δ | p_one | verdict |
|---|---:|---:|:--:|---:|---:|:--|
| empirical prior + backoff (incumbent) | 0.8630 | 0.8553 | — | — | — | reference |
| uniform prior + backoff | 0.8633 | 0.8544 | 6/8 | +0.0003 | 0.218 | not significant |
| empirical prior + uniform Lidstone | 0.8625 | 0.8549 | 0/8 | −0.0005 | 1.7e-06 | **backoff is better** |
| uniform prior + uniform Lidstone | 0.8629 | 0.8540 | 4/8 | −0.0001 | 0.342 | not significant |

Two honest conclusions:

- **Yes, the class-conditional-marginal backoff still beats plain uniform Lidstone under M6**
  (0.8630 vs 0.8625; the Lidstone variant is *worse* with p_one = 1.7e-06). The backoff
  carries real information for the nodes with a feature parent.
- **The class prior does not interact with the DAG.** Uniform vs empirical is +0.03 pt at
  p = 0.22 — a wash, consistent with the near-uniform MNIST class balance. **Keep the
  incumbent estimator unchanged** (α=1.0, empirical prior, class-conditional backoff).

---

## 1. NEW CHAMPION — full specification

### 1.1 What changes (minimal, fully explainable)

**Only two of the twelve feature definitions change**, and only in their coordinate frame.
The DAG, feature count, cardinalities, discretisation rule and smoothing are all M6's.

| slot | M6 feature | champion feature | change |
|---|---|---|---|
| 0–4, 6 | absolute central-window density | unchanged | — |
| **5** | `density_middle_right` (fixed window rows 12..17, cols 17..22) | **`bbox_density_middle_right`** (middle third × right third of the ink bbox) | absolute → content-relative |
| **7** | `density_bottom_right` (fixed window rows 17..22, cols 17..22) | **`bbox_density_bottom_right`** (bottom third × right third of the ink bbox) | absolute → content-relative |
| 8–11 | `euler_topology`, `vertical_runs`, `aspect_ratio`, `vertical_dominance` | unchanged | — |

New features: one continuous statistic each (band ink density), discretised to **5 states**
by `zero_quantile`; semantic = "ink density of the middle-right / bottom-right ninth of the
glyph, measured in the digit's own bounding box so writing position and size are normalised
out". All 12 cardinalities are in {4, 5} ⊂ [2, 5].

### 1.2 DAG — unchanged from M6 (11 feature–feature edges)

```
FEATURE_PARENTS = (7, 0, 1, 8, 3, 8, 7, 5, -1, 8, 9, 9)
```
(0=tl, 1=tc, 2=tr, 3=ml, 4=mc, 5=bbox_mr, 6=bl, 7=bbox_br, 8=euler, 9=vruns, 10=aspect, 11=vdom)

Edge list with per-edge semantics:

| # | edge | semantic |
|--:|---|---|
| 1 | `br → tl` | diagonal ink balance: open-top digits (2, 7) and looped/closed digits (0, 6, 8) trade ink across the anti-diagonal; given Y it remains informative. **Most load-bearing single edge** (the critic measured −5.7 pt on removal). |
| 2 | `tl → tc` | top-band left→right reading continuity (the same stroke crosses the boundary). |
| 3 | `tc → tr` | top-band row continuity. |
| 4 | `euler → ml` | global topology conditions the mid-left band: whether the middle band is cut (2, 3), forms a loop (3) or is a crossing stroke (5) is exactly what the topology invariant encodes and a regional mean cannot. |
| 5 | `ml → mc` | middle-band row continuity. |
| 6 | `euler → mr` | same topology argument for the mid-right band (this is now the **bbox** middle-right density, so topology conditions the *shape-normalised* mid-right density — a strictly cleaner version of the M6 edge). |
| 7 | `br → bl` | bottom-band right→left return stroke (incumbent). |
| 8 | `br ← mr` | the right column as a continuity edge (incumbent); now linking bbox mid-right into bbox bottom-right. |
| 9 | `euler → vruns` | topology constrains stroke fragmentation (looped digits are more coherent). |
| 10 | `vruns → aspect` | fragmentation plus extent determines the observed aspect ratio. |
| 11 | `vruns → vdom` | fragmentation determines gradient directionality. |

Note the pleasing interaction: M6's `euler → mr` and `br ← mr` chain now runs over
bbox-normalised densities, so the topology→shape conditioning is no longer contaminated by
absolute writing position. Acyclic witness (constructive, every arc points rightward under
this order): `euler, vruns, vdom, aspect, mr, bl, br, tl, tc, tr, ml, mc`. Kahn confirms
12/12 (below).

### 1.3 Gate compliance

| requirement | champion |
|---|---|
| 8-seed adoption decision | ✓ all 8 seeds (1322, 7, 2024, 1, 42, 99, 123, 555) |
| pooled McNemar p < 0.05 after Bonferroni | ✓ pooled held-out one-sided p = 4.0e-53; Bonferroni m = 92 → **3.7e-51** |
| positive on ≥6 seeds | ✓ 8/8 |
| multiplicity-corrected p reported explicitly | ✓ (m = 28 own family; m = 92 whole idea-A family) |

`check_bn_bounds(FEATURE_NAMES, FEATURE_CARDINALITIES, FEATURE_PARENTS)` →
**`bayes_net bounds: OK (12 features, 11 feature edges)`**.
Cardinalities `(5,5,5,5,5,5,5,5,4,5,4,4)` all in [2,5]; 11 ≥ 2 feature edges; ≤2 parents each
(Y + ≤1 feature); acyclic; no hidden variables; hand-authored; every CPT smoothed.

### 1.4 8-seed head-to-head, real modules end-to-end (seed-holdout discipline)

`round2_best.BayesNetClassifier(alpha=1.0)` vs `builder_c_dag_m6.BayesNetClassifier(alpha=1.0)`,
each fit on the full 49,999-row training half of its own split:

| seed | M6 | **champion** | margin | McNemar b/c | two-sided p |
|---:|---:|---:|---:|---|---:|
| 1322 (search) | 0.8686 | **0.8820** | +0.0134 | 379/245 | 9.1e-08 |
| 7 | 0.8553 | **0.8704** | +0.0151 | 425/274 | 1.2e-08 |
| 2024 | 0.8662 | **0.8815** | +0.0153 | 408/255 | 3.1e-09 |
| 1 | 0.8641 | **0.8783** | +0.0142 | 386/244 | 1.7e-08 |
| 42 | 0.8609 | **0.8740** | +0.0131 | 427/296 | 1.3e-06 |
| 99 | 0.8678 | **0.8826** | +0.0148 | 414/266 | 1.5e-08 |
| 123 | 0.8595 | **0.8768** | +0.0173 | 428/255 | 3.7e-11 |
| 555 | 0.8616 | **0.8764** | +0.0148 | 388/240 | 3.8e-09 |

- **8-seed mean: M6 0.8630 → champion 0.8778 (+1.47 pt); min 0.8704; positive 8/8.**
- Pooled all-8: b = 3255, c = 2075, p_one = 1.9e-59. Pooled held-out 7: p_one = 4.0e-53.
- Seed 1322 paired bootstrap Δ: **+1.34 pt, 95 % CI [+0.85, +1.83] pt**.
- Macro F1 (seed 1322): M6 0.8661 → **0.8802**; total errors 1314 → **1180**.
- Top confusions (seed 1322): M6 `5→3` 110, `3→2` 75, `3→5` 66, `2→3` 59, `5→2` 54 →
  champion `5→3` **90**, `3→2` **61**, `4→9` 54, `2→3` **46**, `9→0` 42. The improvement is
  concentrated exactly where M6 was weakest (the 5/3, 3/2, 2/3 middle-band cluster), which
  matches the A.3 mechanism: the shape-normalised middle/bottom-right densities resolve
  those digits' right-hand contours independently of how they were written.
- **Unit tests:** the real `assignment2/tests/test_models.py` (9 tests) run green against
  the champion module (monkey-patched in-process — `assignment2/bayes_net.py` is untouched
  on disk).

### 1.5 Honest caveats

- The champion is **not** the global optimum of the bbox idea: k = 5 scores 0.8803
  (+0.25 pt more). It was rejected because it exceeds the mandated 1–2 feature scope and
  buys little; if the main session prefers raw accuracy over minimal change, k = 5 is
  available (`bbox_tr+ml+mr+bl+br`, p_bonf = 6.7e-43).
- As with M6, the exact identity of the two cells is *one* good choice, not uniquely
  optimal: the k = 2 family has 28 members and several clear the gate (table A.4). The
  defensible claim is **"measure the mid-right and bottom-right glyph densities in the
  ink bounding box instead of a fixed window"**, and more generally "normalise writing
  position/scale out of the right-hand contour densities".
- Test-set accuracy is not reported (never read), per the protocol.
- The A.6 `quantile`-discretiser result shows the champion's gain depends on the zero-value
  handling inherited from the incumbent; that is a documented property, not a hidden knob.

---

## 2. Final verdict

**NEW CHAMPION** — adopt `candidates/round2_best.py`:

- `FEATURE_NAMES` slots 5 and 7 become `bbox_density_middle_right` / `bbox_density_bottom_right`
  (bbox-normalised, 5 states, `zero_quantile`); all other 10 features, all cardinalities,
  the DAG `(7, 0, 1, 8, 3, 8, 7, 5, -1, 8, 9, 9)` and the smoothing (α=1.0, class-conditional
  marginal backoff, empirical prior) are unchanged from M6.
- 8-seed mean **0.8778** vs M6 0.8630 (**+1.47 pt**), positive on **8/8** seeds, min 0.8704.
- Pooled held-out McNemar **p = 4.0e-53**; Bonferroni m = 28 → **1.1e-51**, m = 92 →
  **3.7e-51**. Both far below 0.05.
- Fully §4.2-compliant; 9/9 unit tests pass; all modelling NumPy-only.

**KEEN NEGATIVE RESULTS (equally valuable):**
- Idea (B) is exhausted on 8 seeds — every named single-edit refinement (including the
  critic's `vdom` 5-state lead) fails the corrected gate; `p_bonf ≥ 0.28` for all of them.
  M6's DAG is saturated and remains the correct structure.
- The class prior does not interact with M6; the class-conditional-marginal backoff **does**
  still beat plain uniform Lidstone (p_one = 1.7e-06). Estimator unchanged.

**Final line: NEW CHAMPION = `round2_best.py` (M6 DAG + bbox-normalised `density_middle_right`
and `density_bottom_right`, α=1.0 class-conditional-marginal backoff, empirical prior).**

---

## Compliance statement

- Only the 60,000 training images were read (`v4_common.load_split` / `build_cache`);
  `t10k-*.gz` was **never** opened.
- No file outside `experiments_v4/bayes_net/` was created or modified.
  `assignment2/bayes_net.py`, `naive_bayes.py`, `main.py`, `report.md` are untouched
  (the unit-test parity run monkey-patches `bayes_net` **in process only** and restores it
  in a `finally` block).
- Hand-designed structures only: every DAG scored is a named member of a curated family or
  a random draw from an explicit null — **no** hill-climbing / score-based / PC / Chow-Liu
  search selected anything. Idea A is a feature *re-definition* family, enumerated over a
  fixed small set of cells, not a search.
- Every feature has 2–5 states (all are 4 or 5). Seeds fixed
  (1322 + 7, 2024, 1, 42, 99, 123, 555); discretisation thresholds fit on training data only.
- All CPTs smoothed; prediction is `argmax_y [log P(y) + Σ_i log P(F_i | Parents(F_i))]`.
- No pgmpy / bnlearn / sklearn or any ready-made BN library; all NumPy.

### Files

| file | role |
|---|---|
| `candidates/round2_best.py` | **the new champion module** (drop-in: same public API as `bayes_net.py`) |
| `candidates/round2_frontier_experiments.py` | reproduces every number here |
| `candidates/_r2_content.py` | bbox-normalised + control feature extraction and caching |
| `candidates/_r2_common.py` | 8-seed runner, pooled McNemar, Bonferroni summariser |
| `candidates/_r2_fitters.py` | `SweepBN2` (selectable prior / CPT scheme) |
| `candidates/_r2_ideaA_screen.py`, `_r2_mi_sem.py` | idea-A screen and the semantics/redundancy analysis |
| `candidates/cache/round2_content.npz`, `round2_controls.npz` | cached continuous features (train only) |
| `candidates/cache/round2_frontier_raw.json`, `r2_frontier_run.log` | raw numbers and console log |
