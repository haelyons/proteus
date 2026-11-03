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

## Spend

A10 compute 2026-11-03: ~53 billed minutes, ~$1.14 (marker test + four phase-1 launches, three of which
failed within 1-3 minutes on job bugs and were terminated by the driver); evening profiling job ~58 minutes, ~$1.25
(the controlling container restarted mid-run; supervision was re-attached and the instance terminated on completion).
Total ~$2.39.
