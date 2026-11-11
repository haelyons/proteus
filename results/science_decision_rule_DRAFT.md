# Science decision rule: DRAFT (not sealed; awaiting the owner's decision)

Drafted 2026-11-11 from `results/ruler_checks_20261111/` (outcome in
`results/ruler_checks_predictions.md`). Nothing here has been applied to any real column source.

## What the checks established

- `share_struct` (any-category corrected structural p <= 0.01) does not separate trained from
  shuffled-weight units at 8M (98.5% against 98.3–98.6%). It cannot rank bases.
- v1.2 categories rate sums of context units as "structural" (20 planted context parts, each
  "context", give "structural" when summed, whether or not they share a state). They cannot rank
  random directions against neurons (problem B).
- `primary_jsd` separates trained from all three shuffled controls, trained higher
  (Cliff's d +0.17 to +0.22). Mixtures never score above their largest part; mixtures of parts
  with different states score far below them (median 0.09–0.27 against 0.73–0.79), mixtures
  sharing a state slightly below (0.35–0.39 against 0.39–0.63).
- `share_residue_mode` separates too, shuffled higher (100% against 96.9%), and falls as parts are
  summed (0.11 at 20 shared-state parts).
- Noise null of the per-column structural test: at most 0.8% on every measured shape.

## Proposed rule

**Primary statistic:** `primary_jsd` at the peak, per column, positive pole, non-insufficient
columns, ruler v1.2 scoring at B = 2000 with the same descriptor tables for every source.

**Sources, per model size:** N = trained neurons (post-GELU hook, `neurons_post`); R = random
directions in the same space (`randdir:512:0`), sampled per layer like N; S1–S3 = post-GELU
neurons of the three shuffled-weight controls. Equal columns per layer for every source (250 per
layer at 8M).

**Calibration (must hold first, at each model size):** d(N, Sc) >= +0.147 with p < 0.01 for each
of the three controls. If it fails, stop and report; the comparison is not read.

**Decision on H1 (neuron basis privileged), per model size:**

| Outcome | Condition on Cliff's d(N, R) (Mann-Whitney p) |
| --- | --- |
| Supports | d >= +0.147, p < 0.01, and d >= 0.5 x the mean of d(N, Sc) (directions sit at least half way to the untrained floor) |
| Weak support | d >= +0.147 and p < 0.01, below half the shuffled gap |
| Against | d < +0.147 or p >= 0.01 |
| Strongly against | d <= -0.147 with p < 0.01 (directions above neurons) |

**H3 (scale):** d(N, R) and d(N, Sc) at 8M, 650M (and 3B if run), each with a bootstrap 95%
interval (columns resampled within source, 2,000 draws, seeded). "Grows" if the 650M interval of
d(N, R) lies above the 8M interval; "shrinks" if below; otherwise "flat".

**H2 (SAE latents):** only if 5c is built: d(SAE, N) with the same thresholds, reported as not
like-for-like (residual-stream hook against MLP hook).

**Secondary, never decisive:** `share_residue_mode` with intervals; v1.2 call counts from one
pooled `calls()` over all sources; median peak log2.

## Points for the owner to decide

1. **Mixture sensitivity is the mechanism, not a flaw.** Random directions are mixtures by
   construction, and `primary_jsd` falls with mixing across states. A lower score for directions is
   what a privileged basis predicts; equal scores are what "neurons are as mixed as random
   combinations" predicts. But directions also average out states shared by many units, so the
   threshold of "half the shuffled gap" is my proposal, not derived from data.
2. **The 0.147 threshold** is the conventional "small" Cliff's delta. The shuffled gap at 8M sets the
   scale (0.17–0.22), so "supports" asks for a gap of comparable size between neurons and directions.
3. **Two shapes are unchecked:** `randdir` (the direction source's own shape) and `sae:0.001`. Their
   builds timed out on CPU (the randdir generator loops over columns in Python). I propose to
   vectorise the randdir generator and rerun those two shapes (about $2–5) before any real direction
   is scored.
4. **Calibration needs new 8M atlases at the post-GELU hook** (trained and three shuffled, plus
   directions, in one forward per model with `--columns neurons_post,randdir:512:0`): about 1 GPU-hour
   in total at 8M.
