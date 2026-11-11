# Spec: comparison scorer (unit 6)

Status: frozen at the commit that adds this file. No real column source is scored for the
comparison until the checks below are recorded. `atlas/ruler.py` stays byte-identical
(sha256 checked by a test against its blob at this commit).

## Why

Two problems with comparing sources by their share of ruler v1.2 calls:

- **A. Call rates depend on what is scored together.** `calls()` applies Benjamini-Hochberg at
  alpha 0.01 over every test it is given. `scripts/score_real_units.py` calls it once per atlas.
  With B = 2000 Monte Carlo draws the smallest p is 1/2001; v1.2 multiplies the primary
  descriptor's p by the number of descriptors scored (3), so the smallest corrected structural p
  is about 0.0015. That passes BH at rank r of m only if 0.0015 <= 0.01 r/m, i.e. if about 15%
  of the tests in the same call sit at that floor. A source's call rate therefore depends on the
  other columns in the same call.
- **B. "Structural association" needs the structural test to pass and no residue mode.** A
  random direction pools many units, so it is less residue-specific and lands in "structural"
  more often than in "context", whether or not the neuron basis is privileged.

## What is computed

Per column, from `score_column` output alone (no other column involved):

- `primary`: the descriptor chosen exactly as `calls()` chooses it: among the descriptors with a
  peak score, the maximum of `(-log10 p, jsd_excess)`.
- `n_desc`: number of descriptors with a peak score.
- `p_struct`: the primary descriptor's p at the peak. `p_struct_corr = min(1, p_struct * n_desc)`
  (v1.2's correction, without BH).
- `primary_jsd`: `jsd_excess` of the primary descriptor at the peak, as `calls()` computes it.
- `p_res`: `aa_given_fine[primary]["p_max"]` at the peak (v1.2's residue test), or `null`.
- `residue_modes`: residues with `p_res[r] < 0.01` and `log2_res[r] >= 1` in
  `aa_given_fine[primary]` (as `calls()` lists them).
- `insufficient`: true when the peak has fewer than 10 clusters or no descriptor score (the
  cases `calls()` reports as "insufficient samples"); then the fields above are `null`.

These must equal what `calls()` writes into `s["primary"]` and `s["primary_jsd"]` for the same
score dicts (it mutates them), which is the oracle for the reimplementation.

Per source:

- **Primary statistic 1:** `share_struct` = number of non-insufficient columns with
  `p_struct_corr <= 0.01` over the number of non-insufficient columns, in any v1.2 category, with
  a Clopper-Pearson 95% interval. The share over all sampled columns is reported beside it.
- **Primary statistic 2:** the distribution of `primary_jsd` over non-insufficient columns:
  n, mean, and quantiles 0.1, 0.25, 0.5, 0.75, 0.9.
- **Secondary:** v1.2 call counts from **one** `calls(..., rule="v1.2")` over the pooled score
  list of every source named in `--pool`, with the pool's membership and size recorded.

Sampling: for each source, the same number of columns per layer, drawn without replacement by a
seeded permutation (`torch.Generator().manual_seed(seed * 7919 + layer)`) of that layer's
columns, positive pole. The Monte Carlo seed of a column is its flat index in its own atlas
(`layer * D + unit`), as in `score_real_units.py`, so a column's score does not depend on what
else is sampled or pooled. The same descriptor `Tables` (same file, same build) is used for every
source in one comparison; its sha256 goes into every output.

## Frozen interface

```
python scripts/compare_sources.py sample  --atlas NAME=ATLAS_PT [NAME=ATLAS_PT ...] \
        --per-layer K --seed S --out SAMPLE_JSON
python scripts/compare_sources.py score   --sample SAMPLE_JSON --source NAME --atlas ATLAS_PT \
        --desc DESC --data DATA --uniref UNIREF [--B 2000] [--workers W] --out SCORES_DIR
python scripts/compare_sources.py summarise --scores SCORES_DIR --pool NAME[,NAME...] --out SUMMARY_JSON
```

Library entry points (`atlas/compare.py`):

```
column_stats(score: dict) -> dict            # the per-column record fields above, without identity
share_ci(k: int, n: int, level=0.95) -> (lo, hi)   # Clopper-Pearson; (0, hi) at k = 0, (lo, 1) at k = n
summarise(records: list[dict]) -> dict       # per-source primary statistics from column records
```

Files:

- `SAMPLE_JSON`: `{"seed": S, "per_layer": K, "sources": {NAME: {"atlas": PATH, "atlas_sha256":
  ..., "L": L, "D": D, "columns": [[layer, unit], ...]}}}`.
- `SCORES_DIR/<NAME>.jsonl`: one record per sampled column, in sample order:
  `{"source", "layer", "unit", "col", "seed", "pole": 0, "insufficient", "n_clusters_peak",
  "primary", "n_desc", "p_struct", "p_struct_corr", "primary_jsd", "p_res", "residue_modes",
  "score_sha256"}`, where `score_sha256` is the sha256 of
  `json.dumps(score, sort_keys=True).encode()` of the full `score_column` output.
- `SCORES_DIR/<NAME>.scores.pt`: the list of full `score_column` outputs, sample order.
- `SCORES_DIR/<NAME>.meta.json`: `B`, `tables_sha256` (sha256 of the descriptor file's bytes),
  `ruler_sha256`, seconds, workers.
- `SUMMARY_JSON`: `{"sources": {NAME: {"n_sampled", "n_scored", "k_struct", "share_struct",
  "share_struct_ci": [lo, hi], "share_struct_all", "jsd": {"n", "mean", "q10", "q25", "q50",
  "q75", "q90"}, "share_residue_mode"}}, "pooled_v12": {"pool": [NAME, ...], "n_columns",
  "calls": {NAME: {CALL: count}}}, "tables_sha256", "ruler_sha256", "B"}`.

Worked example (numbers for illustration of the arithmetic only). Source `trained` with 4
sampled columns whose records are:

| col | insufficient | p_struct | n_desc | p_struct_corr | primary_jsd |
| --- | --- | --- | --- | --- | --- |
| 10 | false | 0.0005 | 3 | 0.0015 | 0.12 |
| 11 | false | 0.004 | 3 | 0.012 | 0.05 |
| 12 | false | 0.002 | 2 | 0.004 | 0.08 |
| 13 | true | null | null | null | null |

`n_sampled = 4`, `n_scored = 3`, `k_struct = 2` (cols 10 and 12), `share_struct = 2/3 =
0.6667`, `share_struct_ci` = Clopper-Pearson for 2 of 3 = [0.0943, 0.9916], `share_struct_all
= 2/4 = 0.5`, `jsd = {"n": 3, "mean": 0.0833, "q50": 0.08, ...}` (quantiles by linear
interpolation, `torch.quantile` default).

## Checks (predictions committed before each is run)

1. **Pooling.** One seeded sample of trained 8M columns scored alone; the same columns pooled
   with an equal shuffled-weight sample; and pooled with three times as many shuffled-weight
   columns. Per-column records (all fields, `score_sha256` included) must be bitwise identical
   across the three; record how far the v1.2 call counts of the trained columns move between the
   three pooled `calls()`.
2. **Planted mixtures** (added to 5a, `scripts/synthetic_columns.py`): columns equal to the sum
   of m in {2, 8, 20} context signals of the `lys_helix` kind, `1[residue == X and state == S]`,
   over distinct residues X, plus noise at the ladder's SNRs; one family with every part sharing
   S = helix, one with parts over different states (helix, strand, loop, cycling). Each part is
   also planted alone. Record v1.2 categories and the primary statistics for mixtures and parts.
3. **Null rate** of `p_struct_corr <= 0.01` on the planted `noise` columns of every 5a shape,
   with its binomial interval.

After these, a draft science decision rule written from the recorded results goes to the owner
before it is sealed.

## Tests (blind; invariants and arithmetic, not file layout)

- `column_stats` agrees with the fields `calls()` writes (`primary`, `primary_jsd`) on random
  and on planted score dicts, and its `p_struct_corr` equals v1.2's corrected structural p.
- Pooling invariance: a column's record is identical whether it is scored and summarised alone or
  among others.
- `share_ci` against known Clopper-Pearson values, including k = 0 and k = n; `summarise` on the
  worked example.
- Sampling: per-layer counts equal K; same seed gives the same sample; different sources'
  samples do not depend on each other.
- `atlas/ruler.py` is byte-identical to its blob at the spec commit.
