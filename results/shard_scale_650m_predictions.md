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

## Addendum before launch (2026-11-11 18:20 UTC)

The first launch attempt found no `gpu_8x_a100` capacity in us-west-1 (nothing launched). The job
is retried every 5 minutes for up to 12 hours, on the first available of `gpu_8x_a100`, then
`gpu_8x_a100_80gb_sxm4`, in us-west-1, us-east-1 or asia-south-1 (the regions whose filesystems
hold the Swiss-Prot cache). Hard cap 4 h: at most $63.68 on the 40 GB type, $89.28 on the 80 GB
type. If the 80 GB type is used (same A100 architecture, faster memory): prediction 5's range
becomes 2,900–4,170 s; every other prediction stands as written.

## Outcome (2026-11-11, `results/shard_scale_650m_20261111/`)

Run `shard-scale-650m-20261111-1817`: `gpu_8x_a100` (8 x A100-SXM4-40GB, driver 570.148.08) in
us-east-1 on `esm-atlas-results`; healthy 18:24:09Z, job done 20:56:14Z, terminated 20:58:27Z
(about 2.6 billed hours, about $41). One earlier launch exited at startup (default `SCRATCH`
parent missing, no measurement; about $2). All numbers below are from `scaling.json`,
`gate_compare.json`, `digest_compare.json`, `single_vs_ref.json`, `merge.json` and `run.json`.

| # | Predicted | Measured | Held? |
| --- | --- | --- | --- |
| 1 | digests equal on 8 GPUs | batches 0 and 1000, every layer, all 8 equal | yes |
| 2 | exact fields bitwise; `sub_sum`, `sub_sq` bitwise; `gram` <= 1e-5, not bitwise | exact fields bitwise at N = 2, 4, 8; `sub_sum` bitwise; `sub_sq` 4.7e-17 (not bitwise); `gram` 2.6e-7 to 4.0e-7 | no (one detail: `sub_sq` not bitwise); gate passes |
| 3 | single run equals the 4 Nov reference's 9 digests | equal (also every merged atlas) | yes |
| 4 | speedup_stream 1.85–2.0 / 3.5–3.95 / 6.0–7.7 | 1.95 / 3.70 / 6.91 | yes |
| 5 | single stream 3,260–4,170 s | 3,637 s | yes |
| 6 | speedup_e2e 1.5–1.95 / 2.4–3.6 / 3.0–5.0 | 1.81 / 3.07 / 4.59 | yes |
| 7 | merge share of wall_8 10–30% | 10.7% (89.8 s of 838 s) | yes |
| 8 | candidates_ratio 1.2–2.2 / 1.6–3.0 / 1.8–3.5 | 1.66 / 2.33 / 2.64 | yes |
| 9 | redo layer-batches: single 30–90, N = 8 180–540 | 58 and 351 (script's prediction: 58 and 360) | yes |
| 10 | merge peak RSS at N = 8 20–45 GB | 40.4 GB (same at N = 2 and 4) | yes |
| 11 | GPU max allocated equal within 5%, < 40 GB | 20.98–21.07 GiB at every N | yes |
| 12 | wait_s / seconds < 2% (N = 1), < 5% (N = 8) | 0.04%, 0.07% (host batch building is 842 s of work per pass, hidden by prefetch) | yes |
| 13 | prefix naive / filtered 1.5–2.2 | 2.00 | yes |
| 14 | steady naive / filtered >= prefix | 1.27 | no |
| 15 | `cmp_prefix` ok | ok | yes |

Where N = 8 loses against 8x: of 838 s, the serial pilot is 200 s (24%) and the merge 90 s (11%);
the shard stream (527 s) is 6.9x faster than the single stream, below the arithmetic 7.6x by the
repeated warm-up redo (351 against 58 layer-batches) and by 2.6x more candidates per shard.
Removing the serial pilot (layers split across GPUs) and folding the merge as a tree would, by
this arithmetic, bring N = 8 to about 6x end to end. Not built; not measured.
