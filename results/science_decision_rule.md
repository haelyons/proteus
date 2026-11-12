# Science decision rule (sealed)

Drafted 2026-11-11 from `results/ruler_checks_20261111/` (outcome in
`results/ruler_checks_predictions.md`). **Sealed 2026-11-11 by the owner's decision, unchanged from the
draft (commit 4f3319f).** Nothing here has been applied to any real column source. Any later change is an
amendment with its own date and reason, committed before the result it would affect is read.

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

## Points the owner decided on (2026-11-11: seal as drafted; both runs below approved)

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

## Amendment 1 (2026-11-12, the owner's decision; before any direction content is read)

**State when amended.** No random-direction atlas, sample, score or digest from any real network has
been read. The first calibration launch produced nothing. The calibration relaunch
(`calib-8m-20261112-1018`, started 10:18Z, before this amendment arrived) was building its third
shuffled control; only its launcher timeline and step lines had reached this machine.

**Reason.** A positive neuron-to-direction gap d(N, R) has two candidate causes: training placed
structure on the neuron basis, or summing dilutes any elementwise-nonlinear unit, trained or not.
The second predicts the same gap in untrained networks.

**New sources RS1–RS3.** Random directions `randdir:512:0` (same seed, so the same directions R_l as
R) in each shuffled-weight network, built in the same forward as that control's `neurons_post`,
sampled per layer like R (250 per layer at 8M) and scored with the same tables and B.

**New statistic.** d(Sc, RSc): Cliff's d of `primary_jsd` between Sc and RSc, Mann-Whitney p, and a
bootstrap 95% interval (columns resampled within source, 2,000 draws, seeded), as for H3. d(N, R)
gets the same interval.

**H1, changed.** "Supports" and "Weak support" are read only if, in addition to their conditions
above, d(N, R) exceeds every d(Sc, RSc) with non-overlapping 95% intervals (the lower bound of
d(N, R) above the upper bound of each d(Sc, RSc)). If the conditions above hold but this one does
not, the outcome is **"gap not attributable to training"**. "Against" and "Strongly against" are
unchanged.

**Order.** Calibration first (unchanged), then d(Sc, RSc) for c = 1, 2, 3, then H1.

**650M.** The same, with RSc built in the same forward as each shuffled control.

**Where RS1–RS3 are built.** The calibration relaunch had started, so it is not changed (its
shuffled runs build `neurons_pre,neurons_post`, and keep only the trained `neurons_post` and
`randdir_512_0` atlases). The comparison job builds RS1–RS3 by rerunning each shuffled control
with `neurons_post,randdir:512:0` in one forward. Its `neurons_post` records for the calibration
sample must equal the calibration job's (`score_sha256`) as a reproduction check. The comparison
job is still not built; its spec comes first. Predictions for d(Sc, RSc):
`results/rs_controls_predictions.md`, sealed in the same commit as this amendment.
