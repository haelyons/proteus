# Results log

## 2026-11-03 - Phase 1 on one A10 (ESM-2 650M, bf16, 40 real Swiss-Prot batches of ~32k rows)

`phase1_a10_20261103/`: `phase1.jsonl` (one record per stage), per-batch JSONL per stage,
`gpu_tests.txt`, `job_log.txt`. Environment: torch 2.7.0, triton 3.3.0, transformers 5.20.2,
HF attention = SDPA. Overheads are relative to the forward pass on the same batches.

| | predicted (user) | predicted (Claude) | observed |
|---|---|---|---|
| forward, s per 32k-row batch | 1.0-1.4 | 0.7-0.9 | 1.31 (27% of A10 bf16 peak vs FLOP estimate) |
| naive per-batch topk | 50-150% | ~100% | 85% in situ; 81% isolated |
| torch filter | 10-20% | ~10% | 38% in situ (steady state); 32% isolated with `nonzero_static`, 18% with `nonzero` |
| Triton filter | 2-5% | 2-3% | 13% in situ (steady state); 6.4% isolated |
| whole atlas, torch | 8-20% | - | 124% steady state; early dense redos 3-30x |
| whole atlas, Triton | - | - | 98% steady state |

Notes and open items:
- Attention is not the forward bottleneck: shortest vs longest Lmax quartile 1260 vs 1333 ms per 32k
  rows (attention 2% vs 11% of FLOPs). Linear layers run at ~34 TFLOP/s on the A10. Unexplained without a profile.
- `nonzero_static` (sync-free) is slower than `nonzero` on CUDA in torch 2.7: 12.6 vs 7.2 ms per [32768, 5120].
- Triton kernel: exact survivor set on GPU; best tile 16x256 at 2.58 ms = 130 GB/s (22% of HBM peak) vs
  0.56 ms one-read prediction; first call incl. compile 123 ms; no atomic blow-up at 16% survivors. Likely
  fix: skip the per-tile scan and atomic when a tile has no hits.
- Whole atlas: attribution unexplained pending a profile. Suspects: fp64 accumulation on the A10, merging
  ~10M reservoir-tail survivors per batch while the q99.9/q99.99 bins are under-filled, the dense redo path.
- Bf16 ties at tau on real layer-16 activations: 1,961 equal vs 19,750 greater.
- Projection: Swiss-Prot forward-only on the A10 ~2.3 h; with Triton top-K ~2.6 h. A100 ~1 h for top-K is an
  extrapolation, not a measurement.

## 2026-11-03 - Candidate-fraction diagnostic (ESM-2 8M, CPU)

`diag_8M_3000/`: identical activations (3,000 proteins, 1.04M residues, 768 random FFN units, K=16)
streamed in four orders; per-batch counts above and exactly at tau. Prediction: K/(N+1).

Median observed/predicted, fp32, by stream position (early <100k tokens, late >600k):

| order | trained early | trained late | shuffled-weights early | shuffled-weights late |
|---|---|---|---|---|
| random tokens | 1.00 | 0.99 | 1.01 | 0.96 |
| global shuffle, bucket in 64-seq windows | 1.45 | 1.04 | 0.40 | 0.08 |
| current loader (bucket all, shuffled batches) | 1.94 | 1.10 | 3.41 | 0.21 |
| length-sorted (drift control) | 1.94 | 1.08 | 0.20 | 0.08 |

- Trained: lift comes mainly from proteins arriving whole (extremes cluster within proteins), decays toward 1.
- Shuffled-weights control: ties are not the cause (fp32 ties ~1e-5 of candidates); heavy right skew
  (pooled ratio 3.5, 11-20% of batches > 3x).
- bf16: strict `>` on quantised values lowers the iid ratio to 0.95-0.97 (trained), 0.87-0.95 (control).

## 2026-11-03 (evening) - A10 profiling: forward pass, overhead decay, seeding (ESM-2 650M, bf16)

`a10_profile_20261103/`: `profile_forward.json` + `kernel_table.txt` (forward profile, determinism, GEMM microbench,
tile sweep), `seeded/` (120 batches, seeded thresholds) and `long/` (620 batches, 19.6M tokens, unseeded) with
per-batch JSONL (paired hook-free forward per batch), `long/profile_batch_600.txt`, `overhead_decay.png`.
`smi_first_3min.csv` covers only the forward-profile window (the local mirror stopped; full log is on the filesystem).

Forward pass (one 32k-row batch, 1.33 s):
- GEMMs run in bf16 (cuBLAS `ampere_bf16_s16816gemm`, CUTLASS bf16). Kernel time per batch: GEMM 517 ms (40%),
  elementwise ~640 ms (50%), attention 92 ms (7%), layer norm 30 ms (2%).
- Elementwise is HF's ESM code: unfused erf-GELU (five kernels over [32k, 5120] per layer, ~225 ms) and rotary
  embeddings in fp32 with casts/cat/neg (~300 ms).
- Power: forward 143 W mean at the 150 W limit, SM clock 1,262 MHz mean (max 1,695); sustained bf16 GEMM 149.7 W,
  1,082 MHz, 70.8 TFLOP/s = 89% of the clock-scaled peak; 56 C max. Power-capped, not thermal. (Throttle-reason
  fields were not recorded: the probe used `-c`, nvidia-smi's compute-mode flag; fixed.)
- Standalone GEMM at model shapes: bf16 69-75 TFLOP/s; TF32 34.5-34.8 (the forward's 34 TFLOP/s matching TF32 is a coincidence).
- Bitwise reproducible on GPU across repeats, for a bucketed batch and a 57%-padding batch (multi-threaded CPU
  SDPA is not, on padded batches).

Compaction kernel after early exit on empty tiles: best tile 8x512, 4 warps: 0.95 ms per [32768, 5120] (late-stream
thresholds, ~19k survivors) vs 2.58 ms before; 2.08 ms at ~30 survivors per column; torch `nonzero` path 6.6 ms.
x33 layers = 2.4% of the forward. Larger tiles exceed shared memory.

Whole-atlas overhead vs paired forward (median per segment):

| batches | 0 | 1-10 | 10-50 | 50-100 | 100-200 | 200-300 | 300-400 | 400-500 | 500-620 |
|---|---|---|---|---|---|---|---|---|---|
| unseeded | 28.1 | 28.7 | 0.40 | 0.35 | 0.33 | 0.325 | 0.314 | 0.316 | 0.311 |
| seeded | 0.57 | 2.23 | 0.40 | 0.35 | 0.35 (100-119) | - | - | - | - |

- The phase-1 124% was warm-up. Steady state (batches 200-619, non-redo): 0.317 = 1.75 s atlas pass vs 1.33 s forward,
  while compaction fell 40x (9.5e-4 -> 2.3e-5): the floor is fixed per-batch work.
- Far-end profile (batch 600): subsample statistics 322 ms (Gram TF32 ~72 ms, sums/compares/gathers), merges 122 ms,
  compaction 33 ms. Spikes every 100 batches (~3.4x) are the fp32->fp64 host Gram flush (~3% average).
- Seeding: pooled overhead over the first 120 batches 2.43 (unseeded) -> 0.64 (seeded). Verification: 0 of 1,543,826
  seeded reservoir bins short, 0 top-K groups short. Sporadic dense redos remain (15 in 620 unseeded, 13 in 120 seeded).
- Top-K candidate ratio vs K/(N+1) at 6-19M tokens: 0.90 (bf16 ties plus decayed clustering lift).
- Checkpoint without Gram: 1.87 GB in 24-26 s; final save with fp64 Gram: 8.79 GB in 114-118 s (~77 MB/s to the
  network filesystem).

## 2026-11-04 - Implementation A/B on one A10 (ESM-2 650M, bf16, seeded, 620-batch plan)

`impl_ab_20261104/`: per-batch JSONL for A and B, kernel tables and per-range trace attributions at
batches 150 (A, B) and 600 (B), `bench_gram.json`, `compare_A_B.json`, `job.log`, launcher log and
timeline. Predictions were committed before launch: `impl_ab_predictions.md` (bf03e11). A = stats v1 +
merge loop, B = stats v2 + merge stacked, same seeds and batches; B resumed from its batch-160
checkpoint to 620. us-west-1 (the us-east-1 A10 had gone), torch 2.7.0, power-capped at 150 W.

| quantity | predicted | observed |
|---|---|---|
| GPU tests (CUDA exactness, both backends) | pass | pass; `_int_mm` used, `mm(out_dtype=)` unavailable (torch 2.7) |
| A vs B at batch 160: top-K values, reservoirs, bin masses, enrichment, sums | bitwise equal | bitwise equal |
| A vs B: top-K token ids | bitwise equal | **1,126,406 of 21.6M differ**, all at identical values (ties; see below) |
| subsample, A (v1), ms/batch | 322 (290-350) | 332.9 |
| subsample, B (v2) | 155 (120-200) | **340.0** |
| v2 bins / enrich | 35-70 / 10-20 | **120.7 / 90.9** |
| Gram (both) | 72 | **99.7 / 107.3** (GEMM 78.2 + `+=` 21.5; v2 also 7.6 of casts) |
| 66-call reduce (43.7 ms at batch 600) | subsample (central) or merge | subsample: v1 `ge.sum(0)`, 44.7 ms |
| merge bulk, loop vs stacked | 45-100 vs 15-30 | 60.8 vs 28.8 |
| idle inside bulk (sync gaps), loop vs stacked | 7-15 vs < 5 | 8.5 vs 0.9 |
| merge total, batch 150 (A / B) | 125-160 / 70-115 | **~77 / ~45** (sum of sub-ranges) |
| merge total, B batch 600 | 60-100 | **34.6** |
| overhead batches 100-160, A median / mean | 0.33-0.35 / 0.35-0.40 | 0.327 / 0.352 |
| same, B | 0.17-0.21 / 0.19-0.24 | **0.309 / 0.334** |
| overhead batches 500-620, B median / mean | 0.15-0.20 / median + < 0.015 | **0.288** / 0.287 |
| Gram flush (every 500 now) | 1.5-2.5 s | ~2.5 s (batch 499, overhead 2.12) |
| redo batches | 19, 27, 32, 33, 37, 45, 111, 113 | 1, 2, 7, 19, 27, 32, 33, 37, 45, 113 in both A and B; none after 113 |
| bench: Gram TF32 / bf16, n=1600, D=5120 | ~2.2 / 1.1-1.3 ms | 2.30 (incl. cast) / 0.96 ms (87 TFLOP/s) |
| bench: enrichment GEMM int8 vs TF32 | - | 0.52 vs 0.63 ms (~7 TOPS at N=144) |
| seed verification at 620 | 0 short | 0 of 1,590,947 seeded bins short; 0 top-K groups short |
| wall, billed | ~56 min, $1.20 | ~56 min, ~$1.20 |

Notes and open items:
- Subsample v2 failed: it replaced [n, D, B] booleans with int64 [D, n] passes (8.2M elements, 65 MB each,
  ~0.27 ms per pass on the A10): `searchsorted` output, -1 / B- arithmetic, base+index, compares for three
  enrichment levels, padded int8 copies. Same bytes, same time. The prediction undercounted the cost of
  int64 passes. The materialisation hypothesis stands; removing it needs a kernel that never writes them.
- The Gram's `gram[l] += A @ B` is a separate read-modify-write of the 5120^2 fp32 accumulator: 21.5 ms
  per batch in both implementations (accumulate in the GEMM instead).
- Top-K token ids under ties depend on arrival order: the Triton compaction reserves output slots with
  atomics, so candidate order varies run to run, and `s > tau` keeps the first tied token. Values are
  exact; ids are one valid choice among equal values. CPU (torch.nonzero, fixed order) was bitwise equal.
  A deterministic definition needs a (value, smallest token id) key and `s >= tau` admission.
- Steady-state mean - median gap: 0.026 on 2026-11-03 (Gram flush every 100), -0.001 here (every 500).
- Merge at batch 600 is 34.6 ms in B; the 122 ms measured on 2026-11-03 (unseeded, loop, other instance)
  is not reproduced by the loop at batch 150 here (~77 ms). Unexplained (batch position, seeding or instance).
- `smi.csv` mirror stopped 3 min into the job, as on 2026-11-03; the file on the filesystem was not checked.
  Mirror read errors are swallowed; unexplained.

## 2026-11-04 - ESM-2 650M over Swiss-Prot on one A100 40 GB (the science pass)

`science_650m_20261104/`: per-batch JSONL (6,143 batches), `run.json`, trace attributions at batches
500 and 3000, equivalence-gate runs and comparison, GPU test logs, job/launcher/supervisor logs.
Predictions committed before launch: `science_650m_predictions.md` (4123e1c). Instance: gpu_1x_a100_sxm4,
asia-south-1 (first with capacity), torch 2.7.0, triton 3.3.0. The atlas (`atlas.pt`, 10.85 GB) and
checkpoints stay on the `esm-atlas-asia-south-1` filesystem; the run is exactly reproducible
(deterministic ties, seeded thresholds from the global stream), so it can be recomputed elsewhere.
Frozen config: K=64, Kr=64, edges at pilot q 0.5/0.9/0.99/0.999/0.9999/0.99999 (pilot 262,144
tokens), seeded, rho_sub 0.05, rho_res 0.01, stats v3, merge stacked, held-out list excluded.

| quantity | predicted | observed |
|---|---|---|
| GPU tests incl. stats/Gram kernels at n=1600, D=5120 | pass | pass (Gram rel err 6.6e-6 vs fp64; 46,245 values on edges, counts exact) |
| gate (60 batches): v1 + loop vs v3 + stacked | bitwise equal; sums < 1e-6 | bitwise equal on every exact field incl. top-K ids and keys; sums 4.5e-9 / 1.1e-8 |
| stats kernel per layer | 0.10-0.25 ms | **1.0-1.1 ms** (in-run, batches 55/500/3000) |
| Gram kernel per layer | 0.20-0.35 ms | **0.77-0.82 ms** (~55 TFLOP/s on the upper triangle) |
| tokens / batches | 194.2M / ~6,090 | 194,177,125 / 6,143 |
| hook-free forward per batch | 0.40-0.55 s | 0.469 s (mean of 246 paired batches) |
| overhead, paired batches excl. redo/profiled/flush/checkpoint, median | 0.10-0.18 | **0.286** (0.273 after batch 3000); mean 0.296 |
| whole-pass atlas time vs forward-only estimate | - | 62.8 vs 48.0 min (0.309) |
| stream wall | 55-75 min | **77.0 min** = forward 48.0 + atlas 14.8 + host batch building 8.4 + checkpoints 3.8 + paired forwards 1.9 |
| pilot / final save | 2-4 min / 2-4 min | 2.6 min / 2.3 min (10.85 GB) |
| checkpoints | ~55 s | 56-59 s (4.10 GB, 4 of them) |
| redo batches | < 1%, early and extreme proteins | 47 (0.8%), all by batch 1,743, layers 0-1; 0.6 min in total |
| seed verification | 0 short | 0 of 1,666,065 seeded bins short; 0 top-K groups short |
| q99.999 bin mass | 0.3e-5 to 3e-5 | not measured (needs atlas.pt) |
| billed | ~95 min, ~$3.2 | ~102 min (11:43-13:25 UTC), ~$3.4 |

Device time per batch at batch 3000 (ms): compaction 19.1, merge 23.2 (bulk 12.3, top 2.8, reservoir
3.9), subsample 80.9 (stats kernel 40.1, Gram 25.3, int8 enrichment GEMM 11.0).

Notes and open items:
- The claim: one A100 40 GB computes the exact atlas over Swiss-Prot (194M tokens, 168,960 units x 2
  poles) in 82 min end to end (pilot 2.6 + stream 77.0 + save 2.3); the forward pass alone is 48 min.
  Known removable costs: host batch building on the critical path (8.4 min; prefetch on a thread), the
  two Triton kernels running 3-5x slower than predicted (stats kernel: 80 programs on 108 SMs walking
  rows serially, now has an optional row split; Gram: untuned tiles; `scripts/tune_stats.py`), paired
  timing forwards (1.9 min).
- Why the kernel predictions missed: I treated both as bandwidth-bound. The stats kernel is latency-
  bound on too few programs with [16, 64, 8] temporaries per step; the Gram reaches ~55 TFLOP/s of the
  A100's 312.
- Redos were 9% of the first 300 batches, all layer 0 (+1): with Kr=64 the top reservoir bins are not
  served from bulk rows, so compaction admits from the q99.99 edge and single extreme proteins overflow
  layer 0's buffer. Added `--cap-frac-early` (larger buffer for layers 0-1) for the next run.
- Gram flushes (every 1,000 batches) took 4.2-4.5 s on this host (2.5 s on the A10 host): host-side
  widening and adding 3.46 GB.
- `smi.csv` stopped after 2.5 min for the third job in a row, now under a different supervisor
  (reattach): the nvidia-smi logger on the instance probably dies; unchecked.
- The launcher was replaced by a detached `infra/reattach.py` after submission (tool calls here time
  out at 2 h); SIGKILL so the launcher's finally-block would not terminate the instance.

## Spend

A10 compute 2026-11-03: ~53 billed minutes, ~$1.14 (marker test + four phase-1 launches, three of which
failed within 1-3 minutes on job bugs and were terminated by the driver); evening profiling job ~58 minutes, ~$1.25
(the controlling container restarted mid-run; supervision was re-attached and the instance terminated on completion).
2026-11-04: implementation A/B, ~56 billed minutes, ~$1.20 (us-west-1; new filesystem `esm-atlas-us-west-1`
holds the dataset copy, HF cache and this run's checkpoints and atlas, ~16 GB). Science pass on an A100 40 GB
(asia-south-1), ~102 billed minutes, ~$3.40 (new filesystem `esm-atlas-asia-south-1`: dataset, HF cache,
checkpoints, the 10.85 GB atlas). Total ~$7.0.
