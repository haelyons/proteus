# Sharded atlas gate, ESM-2 8M on CPU: predictions

Written 2026-11-11 before the gate has been run on any implementation. Unit tests on small
runs do not count as this measurement; the gate below does.

## Measurement

`SMOKE`-scale but on the whole test set: ESM-2 8M, CPU, fp32, one thread, 500 Swiss-Prot
sequences (`data/swissprot_test500.pt`, UniProt search, first 500 reviewed entries),
`--tokens-per-batch 4096 --pilot-tokens 16384 --K 64 --Kr 16 --seed-thresholds --seed 0
--gram-flush-every 10`. One pilot artifact; one single run from it; shard runs for N = 2, 4, 8
from it; `merge_atlas.py`; `compare_atlas.py merged single`. All numbers come from
`gate_compare.json`, `merge.json`, `run.json` and `batches.jsonl`.

## Predictions

| # | Quantity | Prediction | Wrong if |
| --- | --- | --- | --- |
| 1 | Exact fields `top_val, top_tok, res_tok, res_val, res_key, sub_bin, enr, ann_tot, edges` and scalars `n_seen, n_sub, gram_rows`, merged vs single | bitwise equal for N = 2, 4, 8 | any field differs in any N |
| 2 | `sub_sum`, `sub_sq` max relative difference | at most 1e-12 for every N | larger |
| 3 | `gram` max relative difference | not bitwise equal for at least one N; between 1e-10 and 1e-5 (fp32 accumulation between flushes, regrouped by sharding); passes `--rtol 1e-5` | bitwise equal for all N, or above 1e-5 |
| 4 | Sum over shards of `stats.compacted` divided by the single run's | at least 1.0 for every N, non-decreasing in N, at most 3.0 at N = 8 (seeds computed for the global stream bound the loss on a short stream) | below 1.0, decreasing, or above 3.0 |
| 5 | `res_seed_limited_bins`, `top_seed_short_groups` of merged vs single | equal (they are functions of the exact fields and the shared seeds) | differ |
| 6 | `merge.json` `peak_rss_bytes` at N = 8 | below 1.5 GB (two states of about 0.2 GB each, concatenation temporaries, the torch runtime) | above |
| 7 | Sum of shard stream seconds divided by the single run's stream seconds | between 0.9 and 1.25 for every N (same batches, same thread count, slightly more candidates) | outside |

Decision: the gate passes at 8M when 1, 2, 3 (as a tolerance pass) and 5 hold. 4, 6 and 7 are
calibration for the 650M predictions and do not gate.

## Outcome of the first run, and a second measurement (added 2026-11-11, before the second run)

First run (config above): predictions 1, 3, 5, 6, 7 held; 2 held more strongly than predicted
(`sub_sum`, `sub_sq` bitwise equal); `gram` max relative difference 4.3e-8 to 5.7e-8; merge peak
RSS 0.83–0.91 GB. Prediction 4 could not be evaluated, and the run exposed a weakness of the
gate itself: in the single run 358 of 390 layer-batches overflowed the candidate buffer and were
redone densely (all 390 in the shards), because on a 245k-residue stream most reservoir bins
never fill and the filter threshold stays at a low bin edge. That gate therefore tested the merge
against the dense path, hardly the filtered path. `stats.compacted` only counts batches that did
not overflow, so its ratio was meaningless there.

Second measurement, same data and procedure, chosen so that reservoirs can fill from the fully
examined 1% of rows: `--K 16 --Kr 4 --edge-q 0.5 0.9 0.99` (other arguments unchanged).

| # | Quantity | Prediction | Wrong if |
| --- | --- | --- | --- |
| 8 | Share of layer-batches redone densely in the single run | below 20% (otherwise this config does not exercise the filter either, and the measurement says so) | 20% or more |
| 9 | Predictions 1, 2 (bitwise or at most 1e-12), 3 (at most 1e-5), 5 | hold for N = 2, 4, 8 | any fails |
| 10 | Over shards, the sum of per-batch candidate counts on batches without redo, per such layer-batch, relative to the single run | at least 1.0 for every N and non-decreasing in N | below 1.0 or decreasing |
| 11 | Sum of shard stream seconds over the single run's | between 0.9 and 1.3 | outside |

Correction before any result: `--edge-q 0.5 0.9 0.99` cannot run (the enrichment levels need at
least five bins; the single run stopped with an IndexError in the first batch, nothing was
measured). The second measurement uses `--edge-q 0.3 0.5 0.7 0.9 0.99` instead. Predictions 8–11
are unchanged.
