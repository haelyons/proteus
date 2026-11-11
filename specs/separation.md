# Spec: separation gate (unit 6 addendum)

Status: frozen at the commit that adds this file. Definitions are those sealed in
`results/ruler_checks_predictions.md` ("Separation gate"); this file fixes the interface.
Unit 6's interface and tests do not change.

## Frozen interface

```
python scripts/separation_gate.py --checks OUT --out SEPARATION_JSON
```

Reads only, under `OUT` (the output of `jobs/ruler_checks/run.sh`):
`pooling/scores/{trained,shuf1,shuf2,shuf3}.jsonl` and `.scores.pt`;
`planted/<SHAPE>/columns.json` and `planted/<SHAPE>/scores/<SHAPE>.jsonl` for every shape
directory present. A planted record's `unit` is its column index in `columns.json` (one layer).

Library (`scripts/separation_gate.py`, importable):

```
mann_whitney(x, y) -> {"U": float, "z": float, "p": float}   # two-sided, normal approximation, tie-corrected, no continuity correction; U counts pairs x>y plus half the ties
cliffs_delta(x, y) -> float                                   # (#(x>y) - #(x<y)) / (len(x) len(y))
ci_separation(a: (lo, hi), b: (lo, hi)) -> int                # +1 if a lo > b hi, -1 if a hi < b lo, else 0
```

Statistics, over non-insufficient columns of each source (as `atlas/compare.summarise`):
`share_struct` (`p_struct_corr <= 0.01`), `share_residue_mode` (non-empty `residue_modes`),
`primary_jsd` (distribution). For a share, separation of trained from control c is
`ci_separation(ci_trained, ci_c)` with Clopper-Pearson 95% intervals
(`atlas.compare.share_ci`). For `primary_jsd` it is `sign(cliffs_delta(trained, c))` when
`p < 0.01` and `|d| >= 0.147`, else 0. A statistic `separates` when the three signs are equal and
non-zero; `direction` is that sign (+1: trained higher).

Mixtures, per shape, columns with `alpha >= 2`: for each `mix_*` type, the statistic over that
type's columns (a share, or the median for `primary_jsd`), and the same over each of its parts
(`part_*` types whose `parts` entry is one of the mixture's parts); `above` = mixture value >
max over its parts. Types or values without non-insufficient columns are `null` and are not
`above`. A statistic `rates_mixtures_above_parts` if any mixture in any shape is `above`.

`qualifies[stat] = separates and not rates_mixtures_above_parts`.

Descriptive: per source, the median over non-insufficient columns of
`max(score["peak"][primary]["log2"])` from `.scores.pt`.

`SEPARATION_JSON` keys:

```
{"stats": {STAT: {"by_source": {SOURCE: {...}}, "vs": {CONTROL: {"sign": -1|0|1, ...}},
                  "separates": bool, "direction": -1|0|1, "rates_mixtures_above_parts": bool}},
 "mixtures": {SHAPE: {MIX_TYPE: {STAT: {"mixture": x, "parts_max": y, "above": bool}}}},
 "qualifies": {STAT: bool},
 "peak_log2_median": {SOURCE: x}}
```
`by_source` holds `{k, n, share, ci}` for shares and `{n, median}` for `primary_jsd`; `vs` holds
`{lo_hi_trained, lo_hi_control, sign}` for shares and `{U, z, p, d, sign}` for `primary_jsd`.

## Worked example

`x = [1, 2, 3]`, `y = [2, 4, 5, 6]`: pairs with x > y: 1 (3 vs 2), ties: 1 (2 vs 2), x < y: 10;
`U = 1 + 0.5 = 1.5`; `cliffs_delta = (1 - 10) / 12 = -0.75`. With n1 = 3, n2 = 4, N = 7 and one
tie group of size 2: mean U = 6, tie-corrected variance = n1 n2 / 12 * ((N + 1) - sum(t^3 - t) /
(N (N - 1))) = 1 * (8 - 6 / 42) = 7.8571, `z = (1.5 - 6) / sqrt(7.8571) = -1.6054`, `p = 0.1084`.

Shares: trained 30 of 1,500 (CI [0.0135, 0.0284]) vs control 60 of 1,500 (CI [0.0307, 0.0512]):
`ci_separation = -1`.

## Tests (blind; arithmetic and invariants)

- `mann_whitney`, `cliffs_delta`, `ci_separation` on the worked example and on cases computed
  independently in the test (exact enumeration for small samples, symmetry `d(x, y) = -d(y, x)`).
- On hand-built `OUT` trees: `separates`, `direction`, `above` and `qualifies` for (a) a statistic
  that separates with mixed signs across controls (does not separate), (b) one that separates with
  equal signs, (c) a mixture above its parts.
