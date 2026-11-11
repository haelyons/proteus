# Scaling benchmark, ESM-2 650M on one gpu_8x_a100: predictions

Written 2026-11-11, before any GPU launch. Job: `jobs/shard_scale/run.sh` (spec `specs/scale.md`),
frozen 650M workload, N = 1, 2, 4, 8 on one host with 8 x A100 40 GB SXM4, us-west-1, data from
`esm-atlas-us-west-1/cache/swissprot_2026_03.pt`.

## Basis

`scripts/predict_scaling.py` on the recorded single-GPU run
(`results/followup_650m_20261104/full/batches.jsonl`, `run.json`; one A100 40 GB SXM4,
us-east-1, 4 Nov). Per-shard stream time is the sum of the recorded
`atlas_pass_s` over the batches the shard owns. The fixed costs passed to it are my estimates,
not measurements: `load 10 s, shard_save 30 s, merge_load 20 s, merge_step 10 s, result 15 s,
save 25 s` (local NVMe, states of about 10 GB). Output kept as `pred650.json` beside the results.

| N | shard stream sums (s) | stream_pred (s) | speedup_stream_pred | wall_pred (s) | speedup_e2e_pred | candidates_ratio_pred | redo layer-batches pred |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3,626 | 3,626 | 1.00 | 3,813 | 1.00 | 1.00 | 58 |
| 2 | 1,797 / 1,829 | 1,829 | 1.98 | 2,121 | 1.80 | 1.66 | 116 |
| 4 | 898–924 | 924 | 3.93 | 1,275 | 2.99 | 2.32 | 228 |
| 8 | 448–475 | 475 | 7.64 | 947 | 4.03 | 2.65 | 360 |

## Predictions

| # | Quantity | Prediction | Wrong if |
| --- | --- | --- | --- |
| 1 | Cross-GPU bitwise: `batch_digest.py` digests of batches 0 and 1000, every layer, on all 8 GPUs | all equal to GPU 0 | any differs |
| 2 | Gate, merged vs single, N = 2, 4, 8 | 9 exact fields and 3 scalars bitwise equal; `sub_sum`, `sub_sq` bitwise equal; `gram` not bitwise, max relative difference at most 1e-5 | any exact field differs, or `gram` above 1e-5 |
| 3 | Single run (N = 1) against the recorded reference's 9 field digests (`ref_run.json`), another physical A100 of the same type | all 9 equal, and `n_seen` = 194,177,125 | any differs (then `n_seen` separates a data difference from a forward difference) |
| 4 | `speedup_stream` measured | N = 2: 1.85–2.0; N = 4: 3.5–3.95; N = 8: 6.0–7.7 (below the arithmetic by warm-up redo and by eight processes sharing one host) | outside |
| 5 | `stream_this_process` of the N = 1 run | 3,260–4,170 s (the recorded run's 4,144 s included 117 s of paired forwards and 228 s of checkpoints, absent here) | outside |
| 6 | `speedup_e2e` measured | N = 2: 1.5–1.95; N = 4: 2.4–3.6; N = 8: 3.0–5.0 | outside |
| 7 | Merge share of `wall_8` (merge step / wall_8) | 10–30% | outside |
| 8 | `candidates_ratio` (sum of shard `compacted` over single) | 1.2–2.2 (N = 2), 1.6–3.0 (N = 4), 1.8–3.5 (N = 8) | outside |
| 9 | Redo layer-batches, single and summed over shards | single 30–90; N = 8: 180–540 | outside |
| 10 | Merge `peak_rss_bytes`, N = 8 | 20–45 GB | outside |
| 11 | `gpu.max_memory_allocated` per process | equal within 5% across N = 1, 2, 4, 8, and below 40 GB | spread above 5% or any out-of-memory |
| 12 | Host batch building: sum of `wait_s` over sum of `seconds` per process | below 2% at N = 1, below 5% at N = 8 | above |
| 13 | Filtered vs naive, prefix (first 60 batches, fresh state): naive `sum(seconds)` over filtered | 1.5–2.2 (recorded 650M prefix: stream 858.5 s / 465.6 s = 1.84) | outside |
| 14 | Filtered vs naive, steady (40 batches from batch 4,500, fresh naive state, seeded filtered) | naive over filtered at least the prefix ratio | below it |
| 15 | `cmp_prefix.json` (filtered vs naive atlas, 60 batches) | ok | not ok |

## Cost

Estimated job time about 3 h (setup and tests 20 min, single 64 min, N = 2 35 min, N = 4 21 min,
N = 8 16 min, naive pairs 20 min). Hard cap 4 h: `infra/lambda_job.py --dry-run` gives
gpu_8x_a100 in us-west-1 at $15.92/h, at most $63.68 billed.
