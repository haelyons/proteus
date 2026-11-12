# 8M comparison (N, R, S1–S3, RS1–RS3; H1): predictions

Written 2026-11-12, before the comparison job is built or run and before any direction content
is read. Job `jobs/compare_8m/run.sh` (`specs/compare_8m.md`); rule `results/science_decision_rule.md`
with amendment 1. Predictions RS-1 to RS-4 for d(Sc, RSc) are sealed separately
(`results/rs_controls_predictions.md`, commit 042e1d5) and stand.

## Reasoning I commit to

- Calibration measured d(N, Sc) = +0.236, +0.214, +0.198 (post-GELU, `primary_jsd`); trained
  neurons' median `primary_jsd` is about 0.19 (pre-GELU, recorded).
- A random direction's peak tokens are those with the largest projection, dominated by the few
  units with the largest activations along it. That makes it a mixture of a few neurons' peaks. The
  planted checks show mixtures of parts with different states lose most of their `primary_jsd`
  (0.09–0.27 against 0.73–0.79). Trained neurons carry different states, so R should score below
  N clearly.
- In shuffled networks both neurons and directions are residue read-outs (RS-1), so d(Sc, RSc)
  stays near zero and does not overlap d(N, R).

## Predictions

| # | Quantity | Prediction | Wrong if |
| --- | --- | --- | --- |
| P1 | d(N, R) | +0.15 to +0.45, p < 1e-6 | outside, or p >= 1e-6 |
| P2 | Width of the bootstrap 95% interval of d(N, R) | 0.05–0.12 | outside |
| P3 | H1 verdict | "supports" | any other outcome |
| P4 | `repro`: `N` records equal the calibration's `trained.jsonl` (`score_sha256`) | 1,500/1,500 | fewer |
| P5 | `repro`: each `S<c>` equals the calibration's `shuf<c>.jsonl` | 1,500/1,500 each (multi-source invariance; same sm_80 architecture whichever A100 memory variant the box is) | fewer for any c |
| P6 | Median `primary_jsd`, R against N | R lower | R equal or higher |
| P7 | `share_residue_mode`, R against N | R lower, Clopper-Pearson intervals not overlapping | R equal or higher, or intervals overlap |
| P8 | Shuffled run step times (two sources each) | 11–16 min each | outside |

If calibration recomputed here failed (it cannot if P4 and P5 hold), H1 would be null and nothing
else would be read.

## Addendum before launch (2026-11-12)

Found while drafting the next spec: `randdir:512:0` regenerated on this machine (fp32, CPU) has
`directions_sha256` 852ed324..., the calibration box recorded 625afb88.... Seeded `randn` plus
column normalisation differs at the last bit across platforms. RS1–RS3 are therefore "the same
directions as R" up to rounding, not bitwise.

| # | Quantity | Prediction | Wrong if |
| --- | --- | --- | --- |
| P9 | `directions_sha256` of RS1–RS3 (the three shuffled runs' `run.json`) | equal to each other; equal to the calibration's 625afb88... if the box reports the same torch version as the calibration box, else not | the three differ from each other |

## Outcome (2026-11-12, `results/compare_8m_20261112/`)

Run `compare-8m-20261112-1313`: one `gpu_1x_a100_sxm4` in us-east-1 on `esm-atlas-results` (an
A100-SXM4-40GB this time), healthy 13:17:09Z, job done 14:45:22Z, terminated 14:48Z (about 1.5
billed hours, about $3.00). Numbers from `compare.json`, `repro.json`, `timeline.json` and the
shuffled runs' `run.json`.

**Verdict on H1 at 8M (sealed rule with amendment 1): "against".** Calibration holds
(d(N, Sc) = +0.236, +0.214, +0.198, all p < 1e-19, recomputed here and identical to the
calibration job). d(N, R) = -0.021, p = 0.33, 95% interval [-0.062, +0.023]: random directions in
the trained post-GELU space score the same as neurons on `primary_jsd` (medians 0.1893 and
0.1892). Every d(Sc, RSc) is near zero too (-0.047, +0.011, -0.028). The trained-versus-shuffled
gap belongs to the representation, not to the neuron basis.

| # | Predicted | Measured | Held? |
| --- | --- | --- | --- |
| P1 | d(N, R) +0.15 to +0.45, p < 1e-6 | -0.021, p = 0.33 | **no** |
| P2 | interval width 0.05–0.12 | 0.085 | yes |
| P3 | "supports" | "against" | **no** |
| P4 | N records 1,500/1,500 equal to the calibration's | 1,500/1,500 | yes |
| P5 | each Sc 1,500/1,500 | 1,500/1,500 each (the calibration box was an 80 GB A100, this one 40 GB) | yes |
| P6 | median `primary_jsd` R below N | R 0.18935, N 0.18920 (equal to three decimals) | **no** |
| P7 | `share_residue_mode` R below N, intervals apart | R 99.4% [98.9, 99.7] above N 96.9% [95.9, 97.7] | **no** (opposite) |
| P8 | shuffled runs 11–16 min | 12.4–12.6 min | yes |
| P9 | RS1–RS3 direction hashes equal to each other, and to the calibration's if the torch build matches | all three 625afb88..., equal to the calibration's | yes |

Secondary: pooled v1.2 calls rate R mostly "context" (1,485 of 1,499) against N's 1,397, and
"structural" 12 against 80. These counts are never decisive (problem B).

As instructed, the 8M comparison line stops here. Nothing at 650M, 3B or with SAE latents is
started without the owner's decision.
