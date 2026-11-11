# Spec: planted columns of each shape, mixtures and noise null (unit 5a, with unit 6's checks)

Status: frozen at the commit that adds this file. Replaces section 5a of `specs/columns.md`.
`atlas/ruler.py` stays byte-identical. Without the new flags, `scripts/synthetic_columns.py`
writes a `synthetic_atlas.pt` and `columns.json` bitwise identical to today's.

## Why

Ruler v1.2 was validated on planted columns of one shape, `z = alpha * signal + N(0, 1)`. The
sources to be compared have other shapes (`specs/columns.md`), and unit 6 adds two questions the
planted set must answer: what the ruler and the per-column statistic do with mixtures of
context jobs (a random direction pools many units), and the per-column statistic's null rate.

## Additions to `scripts/synthetic_columns.py`

New arguments (defaults reproduce today's output):

| Argument | Default | Meaning |
| --- | --- | --- |
| `--shape` | `gauss` | `gauss`, `gelu`, `gelu_off`, `randdir`, `sae` (table below) |
| `--sae-rate R` | `0.01` | firing rate of the `sae` shape on noise-only columns |
| `--mixtures` | off | add the mixture and part columns below |
| `--noise-columns N` | `0` | add N extra pure-noise columns (type `noise`, alpha 0) |

Shapes. `z` is computed exactly as today (same generator, same draws, same order). Any extra
randomness comes from a second generator, `torch.Generator().manual_seed(nseed(p) + (1 << 40))`,
drawn after `z`.

| Shape | Stored column |
| --- | --- |
| `gauss` | `z` |
| `gelu` | `gelu(z)` (`torch.nn.functional.gelu`, exact erf form) |
| `gelu_off` | `gelu(z - 2)` |
| `randdir` | `(alpha * signal + sum_{j<64} w_cj * gelu(n_j) - mu) / s`, with `n_j ~ N(0, 1)` independent per residue and column from the second generator, `w_c` a unit vector per column from `torch.Generator().manual_seed(1_000_000 + c)`, and `mu`, `s` the mean and standard deviation of `sum_j w_cj gelu(n_j)` for that `w_c` (exact: `mu = m1 * sum_j w_cj`, `s^2 = v1 * sum_j w_cj^2 = v1`, where `m1 = E gelu(N) = 1 / (2 sqrt(pi))` and `v1 = Var gelu(N)` by Gauss-Hermite quadrature with 200 nodes) |
| `sae` | `relu(z - t)`, `t = ndtri(1 - R)` (`torch.special.ndtri`) |

For `noise` columns `alpha * signal` is 0 in every shape.

Mixtures (with `--mixtures`). `RES_ORDER = "KEDRLAQSTNVIGFYHMWCP"`. Part j of family `same` is
`(RES_ORDER[j], helix)`; part j of family `diff` is `(RES_ORDER[j], [helix, strand, loop][j %
3])`, with `state5` codes loop 0, helix 1, strand 2. A part's signal is
`1[residue == X and state5 == S]`. Column types:

- `mix_same_<m>`, `mix_diff_<m>` for m in 2, 8, 20: signal = sum of parts j < m of that family;
- `part_same_<j>`, `part_diff_<j>` for j < 20: that part alone.

All are planted at every alpha of the ladder with the same replicates as the other types, and
appended after today's columns in this order: mixtures (same 2, 8, 20, then diff 2, 8, 20),
parts (same 0..19, then diff 0..19), extra noise columns; within a type, alpha then replicate,
as today.

Outputs: each `columns.json` entry keeps `col, type, alpha, rep`, and mixture and part entries add
`"parts": [[X, S], ...]` (S as the `state5` code). `synthetic_atlas.pt` adds `shape`, `sae_rate`,
`sae_threshold`, `randdir_mu` (per column, or null) and `randdir_s` only when a new flag is set.

`scripts/eval_synthetic.py`: types without an `EXPECT` entry are not graded; their calls appear in
`calls_by_type_alpha` as today and in a new `ungraded` count table. Default output for today's
types is unchanged.

## Worked example (signals only)

One protein, sequence `KEAK`, `state5 = [1, 1, 0, 1]` (helix, helix, loop, helix).

- `part_same_0` = (K, helix): signal `[1, 0, 0, 1]`; `part_same_1` = (E, helix): `[0, 1, 0, 0]`.
- `mix_same_2`: `[1, 1, 0, 1]`.
- `part_diff_1` = (E, strand): `[0, 0, 0, 0]`; `mix_diff_2` = (K, helix) + (E, strand): `[1, 0, 0, 1]`.
- `sae` with R = 0.01: `t = ndtri(0.99) = 2.3263`; a residue with `z = 3.0` stores `0.6737`,
  with `z = 1.0` stores `0`.

## Measurement (after tests pass; predictions committed first)

Needs the descriptor subset (`desc_subset/`) and the Swiss-Prot cache on the Lambda filesystem,
so it runs on an instance (owner's approval first). For each shape (`sae` at R = 0.1, 0.01,
0.001): `synthetic_columns.py --mixtures --noise-columns 500` with the design proteins and seeds
of `results/synthetic_ruler_20261105`, then `run_ruler.py --rule v1.2`, `eval_synthetic.py`, and
`compare_sources.py sample/score/summarise` on the same atlas (all columns: one layer, per-layer =
C). Recorded per shape:

1. v1.2 gate per type against `gauss` (`specs/columns.md` 5a criteria).
2. For mixtures and their parts: v1.2 category, `p_struct_corr`, `primary_jsd`, residue modes.
3. Null rate of `p_struct_corr <= 0.01` over the noise columns with its Clopper-Pearson interval.

## Tests (blind; invariants and arithmetic)

- Default arguments: output bitwise identical to the code at the spec commit (run both on a small
  synthetic descriptor set built in the test).
- `gelu`, `gelu_off`, `sae`: stored values equal the stated transform of the `gauss` run's values
  for the same column (same seeds), elementwise, to fp32 rounding.
- `sae` on noise columns: fraction of non-zero values within a binomial 99.9% interval of R.
- `randdir`: on noise columns, mean within 0.05 and standard deviation within 0.05 of 0 and 1
  over at least 50,000 values; for every alpha, `column - alpha * signal / s` has mean within
  0.05 of 0 and standard deviation within 0.05 of 1 (the noise part does not depend on alpha).
- Mixtures: a mixture's signal equals the sum of its parts' signals; `mix_*_2` parts are the
  first two of `mix_*_8`'s, which are the first eight of `mix_*_20`'s; worked example above.
- `--noise-columns N` adds exactly N columns of type `noise`.
- `atlas/ruler.py` byte-identical to its blob at the spec commit.
