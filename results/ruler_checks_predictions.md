# Ruler checks (5a shapes, mixtures, noise null; unit 6 pooling): predictions

Written 2026-11-11, before any of these columns exist on the real descriptor set. Job:
`jobs/ruler_checks/run.sh` (`specs/ruler_checks.md`), design proteins of measurement #2 (60,000
proteins, 20,557,222 residues), ruler v1.2 at B = 2000 unchanged. Baseline: the recorded v1.2
gates on three draws (`results/synthetic_ruler_20261105/{design,draw1,draw2}/gate_v1.2.json`):
every type 12/12 except narrow 8–10/12, hydro_win 11–12/12, trp 11–12/12; noise and every
alpha = 0: 42/42.

Note: adding columns changes `C` in `randn(n, C)`, so the base types get a fresh noise draw here,
not the design draw. Their gates are compared with the three recorded draws' range.

## Reasoning I commit to

The atlas is rank-based. A transform that is strictly increasing over the part of the column's
range at or above the q90 edge leaves which tokens fall in the q90 to peak bins unchanged, and
the reservoirs keep tokens by hash key, so it leaves the scored samples unchanged.
- `gelu` is increasing above -0.75 and `gelu_off` above 1.25 (z > q0.894), so both keep q90 and up.
- `sae` maps every value below `t` to 0. At R = 0.1 (`t` = q0.9) the q90 edge is still the firing
  boundary. At R = 0.01 and 0.001 the q90 (and at 0.001 the q99) bins collapse onto the zero
  mass: those levels lose their samples, and the peak and q99.99 levels are unchanged.
- `randdir` changes the noise shape at equal SNR. Its calls should differ from `gauss` only as two
  noise draws differ.

## Predictions

| # | Quantity | Prediction | Wrong if |
| --- | --- | --- | --- |
| 1 | `gauss`: v1.2 gate | within the recorded draws' range for every graded type; noise 100% "none" | any type below its recorded minimum by more than 1 column, or any noise column called |
| 2 | `gelu`, `gelu_off`, `sae:0.1` vs `gauss` | per-column `score_sha256` equal for at least 99% of columns; v1.2 gate identical | below 99% |
| 3 | `sae:0.01`, `sae:0.001` vs `gauss` | peak-level calls (category, primary, modes) identical for at least 99% of columns; the persistence label differs for at least 25% of the columns that have one ("peak only" more often) | category identity below 99%, or persistence unchanged |
| 4 | `randdir` | gate within the recorded draws' range | as #1 |
| 5 | Noise null rate of `p_struct_corr <= 0.01`, every shape (n = 518) | at most 2%; the 95% upper bound below 3% | above |
| 6 | Parts (`part_same_*`, `part_diff_*`), `gauss`, alpha >= 2 | "context association" for at least 90%; `p_struct_corr <= 0.01` for at least 95% | below |
| 7 | `mix_same_2`, alpha >= 2 | "context association" (residue modes K and E) in at least 5 of 6 | fewer |
| 8 | `mix_same_8`, alpha >= 2 | "structural association" in at least 4 of 6 (each residue's share of the peak is too diluted for log2 >= 1) | fewer |
| 9 | `mix_same_20`, alpha >= 2 | "structural association", one mode (helix), 6 of 6 | any other |
| 10 | `mix_diff_2`, `mix_diff_8`, `mix_diff_20`, alpha >= 2 | "context association" in at least 5 of 6 each (residues stay specific given the state); several modes for `_8` and `_20` | fewer |
| 11 | Every mixture and part, alpha >= 2 | `p_struct_corr <= 0.01` for at least 95%: the primary statistic does not depend on the category | below |
| 12 | Pooling: `records_identical` (two scorings of the trained sample, different worker counts) | true | false |
| 13 | Pooling: trained `share_struct` across the three pools | identical | differs |
| 14 | Pooling: trained v1.2 call counts, alone vs pooled with 1x and 3x shuffled | change by at most 5% of the trained columns per category; "none" does not increase with more shuffled columns | outside |

Predictions 8 and 9 are problem B in miniature: sums of context units, each of them called
"context", come out "structural" under v1.2 while the per-column primary statistic stays the same.
If 8 and 9 hold, v1.2 categories are not usable to compare random directions with neurons, and the
comparison rests on the primary statistic.

## Separation gate (unit 6 addendum, owner's request 2026-11-11)

Computed after the job from its saved per-column records (`scores/<source>.jsonl`; trained and
shuf1..3, 250 columns per layer, 1,500 per source), by a separate analysis step. Unit 6's
interface and tests do not change.

Definitions (fixed now):

- **Separates trained from control c.** Share statistics (`share_struct`, `share_residue_mode`):
  the Clopper-Pearson 95% intervals of trained and of c do not overlap. `primary_jsd`: two-sided
  Mann-Whitney U, p < 0.01 (normal approximation with tie correction), and Cliff's delta |d| >= 0.147.
- **Separates** = separates trained from each of shuf1, shuf2 and shuf3, with the same sign.
  The direction is recorded.
- **Does not rate mixtures above parts** (check 2, every scored shape, alpha >= 2 pooled over alphas
  and replicates): for each mixture type, its value is at most the largest value among its parts.
  Shares are computed over that type's columns; `primary_jsd` uses the median.
- **Qualifies** = separates and does not rate mixtures above parts.
- Descriptive only: the median of the largest peak log2 ratio of the primary descriptor, per source.

| # | Quantity | Prediction | Wrong if |
| --- | --- | --- | --- |
| 15 | `share_struct`, trained vs shuffled | does not separate (all three intervals overlap); every source between 95% and 100% | separates |
| 16 | `primary_jsd` | separates, shuffled higher than trained (Cliff's d of trained vs each control <= -0.147) | no separation, or trained higher |
| 17 | `share_residue_mode` | separates, shuffled higher than trained | no separation, or trained higher |
| 18 | Mixtures vs parts | no statistic rates any mixture above its largest part | a mixture above |
| 19 | Peak log2 median (descriptive) | trained about 1.2, shuffled about 1.6, as in `results/real_units_20261105` | trained at or above shuffled |

If 16 and 17 hold, both qualifying statistics rate the untrained network as more concentrated
than the trained one. They then measure distance from a shuffled-weight network, not "more
structure", and the decision rule has to be phrased that way. That is for the owner to decide
before sealing.

## Outcome (2026-11-11, `results/ruler_checks_20261111/`)

Two A10 boxes in us-east-1 read the same inputs (sha256 equal, `inputs.json`): planted shapes on
`ruler-checks-20261111-1819` (4 h cap reached; `randdir` and `sae:0.001` were still building at
the deadline and are **not measured**), pooling on `ruler-pooling-20261111-1950` (complete). An
earlier checks launch failed at its inputs step (wrong recorded paths for two shuffled atlases).
Spend for the three A10 launches: about $6.

| # | Measured | Held? |
| --- | --- | --- |
| 1 | gauss gate: every type within the recorded range (narrow 8/12); 2 of 542 noise and alpha = 0 columns called | no (2 noise columns called) |
| 2 | gelu: 1,490/1,490 records identical to gauss, gate identical. gelu_off: 78/1,490 identical (GELU maps the far-left tail just above its minimum, into the q90/q99 bins), gate identical except one noise column. sae:0.1: 1,429/1,490 identical, gate identical | no (gelu only) |
| 3 | sae:0.01: peak calls identical for 98.3%; persistence differs for 18.1% of the columns that have one; "peak only" 194 -> 191. sae:0.001 not measured | no |
| 4 | randdir not measured (build timed out) | not measured |
| 5 | noise null rate of `p_struct_corr <= 0.01`: 0.39%, 0.39%, 0.77%, 0.39%, 0.58% (gauss, gelu, gelu_off, sae:0.1, sae:0.01); largest 95% upper bound 1.97% | yes |
| 6 | parts, alpha >= 2: 360/360 context, 360/360 at `p_struct_corr <= 0.01` | yes |
| 7 | `mix_same_2`: context 9/9, residue modes E, K | yes |
| 8 | `mix_same_8`: structural 8/9 | yes |
| 9 | `mix_same_20`: structural 9/9, one mode | yes |
| 10 | `mix_diff_2` context 9/9 (several modes 9/9); `mix_diff_8` context 9/9 (several modes 2/9); `mix_diff_20` structural 8/9 | no |
| 11 | every mixture and part, alpha >= 2: `p_struct_corr <= 0.01` 100% | yes |
| 12 | `records_identical` true | yes |
| 13 | trained `share_struct` 0.9847 in all three pools | yes |
| 14 | trained v1.2 counts identical in all three pools (context 1,400, structural 76, composition 21, none 2, insufficient 1) | yes |
| 15 | `share_struct` 98.47% trained vs 98.46 / 98.26 / 98.59%: intervals overlap | yes |
| 16 | `primary_jsd` separates, **trained higher**: medians 0.189 vs 0.140 / 0.147 / 0.156, Cliff's d +0.225 / +0.202 / +0.167, p < 1e-14 | no (direction) |
| 17 | `share_residue_mode` separates, shuffled higher: 96.9% vs 100 / 100 / 100% | yes |
| 18 | no statistic rates a mixture above its largest part (all five shapes) | yes |
| 19 | peak log2 median 1.205 trained vs 1.565 / 1.602 / 1.669 | yes |

Qualifying statistics (separate trained from shuffled, do not rate mixtures above parts, on the
five measured shapes): **`primary_jsd`** (trained higher) and **`share_residue_mode`** (shuffled
higher). `share_struct` does not qualify. The randdir and sae:0.001 shapes remain to be checked.

## Outcome addendum: `randdir` and `sae:0.001` (2026-11-12, `results/ruler_checks_rerun_20261112/`)

Measured on the rerun (`results/calib_8m_predictions.md`, outcome section). #3 for `sae:0.001`: no
(98.26% < 99%; persistence differs 40.5%, "peak only" 194 -> 106). #4: no (`randdir` gate:
`narrow` 3/12, 5 of 542 noise and alpha = 0 columns called). #5: yes (1.16% and 0.58%; upper
bounds 2.50% and 1.68%). #18: yes on all seven shapes. The qualifying statistics are unchanged.
