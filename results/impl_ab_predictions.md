# impl_ab: predictions (written 2026-11-04, before the run)

Job: `jobs/impl_ab` on one A10 (us-east-1, Lambda Stack torch 2.7.0, power cap 150 W), ESM-2 650M,
bf16, Triton compaction, seeded, 620-batch plan. A = stats v1 + merge loop (reference), B = stats v2 +
merge stacked; both stop at batch 160 with a checkpoint, then B resumes to 620. Profiles at batch 150
(A and B, same state) and 600 (B). Baseline: `a10_profile_20261103/long` (unseeded, v1 + loop).

Units: per batch of 32k tokens, all 33 layers, ms of device span (what key_averages calls CUDA total),
plus busy time from the trace attribution. Overhead = atlas pass / hook-free forward - 1.

## Exactness

| check | prediction |
|---|---|
| GPU tests (synthetic exactness on CUDA, both backends) | pass; `_int_mm` path taken; `mm(out_dtype=)` unavailable on torch 2.7 so the Gram takes the TF32 fallback |
| A vs B checkpoints at batch 160 | top-K, reservoir, bin masses, enrichment, n_seen, n_sub bitwise equal; sub_sum / sub_sq bitwise equal (same moment code) |
| seed verification at 620 | 0 short bins |

## Subsample statistics (hypothesis 1)

| range | A (v1) | B (v2) | note |
|---|---|---|---|
| atlas.subsample (span) | 322 (290-350) | 155 (120-200) | batch-independent work; v1 anchored on the batch-600 measurement |
| sub.gram | 72 | 72 | torch 2.7 has no bf16-in/fp32-out mm: same TF32 GEMM on fp32 copies |
| sub.moments | 10 | 10 | identical code |
| sub.bins (+enrich in v1) | 200-230 | bins 35-70, enrich 10-20 | v1: [n, D, B] booleans, permute, float copies |
| gather + cast (in subsample, outside sub.*) | ~10 | ~10 | |

- bench_gram at n=1600, D=5120: Gram TF32 on fp32 ~2.2 ms; bf16 GEMM ~1.1-1.3 ms (the cost with
  out_dtype on torch >= 2.8, or a Triton bf16 GEMM with fp32 accumulator); one read of Hs ~0.05-0.1 ms.
- The user's ~2 ms per layer (~66 ms per batch) is not reached by v2: the Gram alone is 2.2 ms per layer
  on this stack (1.1 in bf16), and v2 still makes several passes over [D, n] int64 intermediates.

## Merge (hypothesis 2)

The 122 ms at batch 600 is a span: it includes GPU idle time behind the 66 host syncs (nonzero), not
only kernels. Candidate inputs at batch 600 are small (154k compacted, 26k top-K, 22k reservoir).

| range | A (loop) batch 150 | B (stacked) batch 150 | B batch 600 |
|---|---|---|---|
| atlas.merge (span) | 125-160 | 70-115 | 60-100 |
| merge.bulk | 45-100 | 15-30 | 15-30 |
| idle inside merge (span - busy) | 7-15 | < 5 | < 5 |
| merge.cand + .top + .res | 15-40 | same as A | 10-30 |

- Syncs themselves: 7-15 ms of the 122. Stacking saves 25-40 ms if the 66-call reduce kernel
  (43.7 ms, 0.66 ms/call) belongs to the subsample's ge.sum(0); 60-80 ms if it is the loop's
  bin_of(...).sum(-1). The batch-600 kernel table cannot tell (both are 66 calls); the trace attribution
  decides. Branch (a) is my central guess.
- Prefilter: keeps all bulk rows in this 620-batch plan (top-bin key ceilings ~3% of the key range,
  above the 1% bulk cutoff); it only bites in the full pass.

## Overhead

| window | A | B | baseline (long, unseeded) |
|---|---|---|---|
| batches 100-160, median | 0.33-0.35 | 0.17-0.21 | 0.337 |
| batches 100-160, mean | 0.35-0.40 | 0.19-0.24 | 0.362 |
| batches 500-620, median | - | 0.15-0.20 | 0.311 |
| batches 500-620, mean | - | median + < 0.015 | 0.337 |

- Mean minus median at steady state was the Gram flush every 100 batches (batches 199, 299, 399, 499,
  599 at overhead 3.4-3.5, about 4.5 s each), not redos (none after batch 113). Now every 500 batches,
  copying fp32 and widening on the host: predicted 1.5-2.5 s per flush.
- Redo batches in the seeded plan: 19, 27, 32, 33, 37, 45, 111, 113 again (data-driven, implementation
  independent), about 2.4 forward-equivalents each.

## Wall time and cost

Boot about 5 min; preflight and GPU tests 2 min; bench 1 min; A 11 min (pilot 1.5 + 160 batches at
~3.4 s); B 10.5 min; compare 1 min; B resume 460 batches at ~2.9 s plus 2 checkpoints and the final
save, 25 min. About 56 min billed at $1.29/h: $1.20 (cap 80 min: at most $1.72).
