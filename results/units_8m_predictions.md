# Reading units by intervention at 8M: predictions

Written 2026-11-12, before the module or the study exists and before any column of the sample has
been read. Spec `specs/units.md` (frozen, f839fb0); job `jobs/units_8m/run.sh`. Populations: 150
trained post-GELU neurons and 150 random directions (25 per layer each), never paired.

## Reasoning I commit to

- **Check 1 (held-out peak tokens match the column's own discovery profile better than each of 20
  others').** Trained neurons' peaks are mostly residue-specific (a residue mode for 96.9% of
  sampled units, pre-GELU), with structural context on top. About 50–150 held-out peak tokens
  estimate the profile well. A neuron fails mainly when one of its 20 rivals shares its top
  residue and context closely. A random direction's peak mixes the few units that dominate its
  projection, so its profile is broader. Broad profiles look alike, so the own-versus-rival margin
  is smaller.
- **Check 2 (direct write predicts steering better than 20 rivals' writes do).** At the last layer
  steering is the direct write plus nothing (no later layer), so both populations pass there.
  Earlier layers add later-layer transformations. I expect no population difference beyond
  noise: the check is about linear read-out, not about the basis.
- **Check 3 (removal at peak exceeds all three controls).** Controls (i), another column at the
  same positions, and (iii), the same column at random positions, change values near the median
  and should rarely compete. The write-matched control (ii) decides the check. It asks whether
  the column's own write matters more at its peak than an equal-size write in another
  same-population direction. I expect that more often for neurons than for directions.

## Predictions

| # | Quantity | Prediction | Wrong if |
| --- | --- | --- | --- |
| U1 | `repro_unit_writes.ok` | true (all 900 cases within the tolerances) | false |
| U2 | Insufficient columns (fewer than 10 held-out peak positions) | at most 5% of each population | above |
| U3 | Check 1 pass rate, neurons | 70–90% | outside |
| U4 | Check 1 pass rate, directions | 30–60% | outside |
| U5 | Check 2 pass rate, neurons | 55–85% | outside |
| U6 | Check 2 pass rate, directions | 50–85% | outside |
| U7 | Check 2 at layer 5, both populations | at least 23 of 25 each | below |
| U8 | Check 3 pass rate (median replacement), neurons | 35–65% | outside |
| U9 | Check 3 pass rate (median replacement), directions | 15–45% | outside |
| U10 | Check 3, controls (i) and (iii) alone (target above each), both populations | at least 85% each | below |
| U11 | Ordering of point estimates, neurons above directions | checks 1 and 3: yes; check 2: within 15 points | any of these fails |

Pass rates are over sufficient columns with Clopper-Pearson 95% intervals (`units.json`). The
predictions are about the rates; no single column's card is predicted.

## Outcome of the first launch (2026-11-12): reproduction gate failed; study not run

Run `units-8m-20261112-1451` (one `gpu_1x_a10`, us-east-1; healthy 14:56:41Z, terminated 15:08Z,
about $0.30). `repro_unit_writes`: 900 cases, 1 bitwise, maximum absolute differences
`dlogp_true` 1.0e-5, `kl` 3.7e-6, `travel_kl` 1.1e-5, `ok` false under the frozen tolerance
(1e-6 absolute). The job stopped before the study, as specified. **U1: no.** U2–U11 not measured.

Diagnosis on this machine (`results/units_8m_repro_20261112/local_compare.json`; the 297
sequences fetched from UniProt): the module equals the legacy `scripts/unit_writes.py` hook bitwise
on all 1,800 effects (900 cases and their base units). Both differ from the 5 Nov record by the
same amounts (maximum 9.6e-6, median 2.8e-7, 133 bitwise). The legacy script no longer reproduces
its own record on today's software stack (torch 2.15 here). The module reproduces the legacy
computation exactly. My tolerance was set below the drift between the recording stack and today's.

## Addendum before the relaunch (2026-11-12; amendment 5c)

| # | Quantity | Prediction | Wrong if |
| --- | --- | --- | --- |
| U1' | Amended gate: module equals the legacy hook bitwise on the box (1,800 effects), both within 2e-5 of the record | `ok` true; 1,800/1,800 bitwise | false, or fewer |

U2–U11 stand as sealed.

## Outcome of the second launch (2026-11-12): U1' held; stopped at `columns`

Run `units-8m-20261112-1825` (one `gpu_1x_a10`, us-east-1; healthy 18:29:45Z, terminated 19:01Z,
about $0.68). Files in `results/units_8m_launch2_20261112/`.

- **U1': yes.** `repro_unit_writes.ok` true; module equals the legacy hook bitwise on all 1,800
  effects; maximum absolute differences from the record are `dlogp_true` 1.01e-5, `kl` 3.7e-6 and
  `travel_kl` 1.07e-5, the same for both. The blind reproduction test passed on the box (1 passed,
  476 s).
- `sample` (2 s) and `heldout_values` (499 s) passed.
- `columns` failed after 7 s. `torch.quantile` rejects inputs above 2^24 elements, and the burial
  cut points cover the whole build. The SMOKE's data is too small to reach the limit. Fix: amendment
  5d (`quantile_linear`, bitwise equal to `torch.quantile` where torch accepts the input).
  U2–U11 were not measured and stand as sealed.
