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

## 2026-11-04 (evening) - Follow-up on one A100 40 GB: batch prefetch and stats-kernel row split

`followup_650m_20261104/`: per-batch JSONL, `run.json` (with per-field sha256 digests), trace
attributions at batches 500 and 3000, gate runs and comparison, row-split sweep, held-out atlas
record, redundancy per bin (`bin_clusters.json`), `ps.log` (nvidia-smi logger every 2 min), the full
`smi.csv`. Predictions committed before launch: `followup_650m_predictions.md` (48fe506). Instance:
gpu_1x_a100_sxm4, us-east-1 (same GPU model as the science pass). Only two changes: batches and their
host-side contexts built on a prefetch thread, and the stats kernel's row split (R automatic).
Atlas copy #2 (10.85 GB) is on `esm-atlas-results` (us-east-1); #1 on `esm-atlas-asia-south-1`.

| quantity | science pass | predicted | observed |
|---|---|---|---|
| stats kernel, ms per batch (in run, batch 3000) | 40.1 | 7-11 | **19.1** |
| stats kernel per layer, sweep at the 650M shape (R = 1 / 2 / 4 / 6 / 8 / 16) | - | - | 1.67 / 1.00 / 0.96 / 0.95 / 0.82 / 0.86 ms |
| mean gap between batches | 0.119 s | 0.005-0.03 s | **0.000 s** |
| make_ctx inside the atlas pass | 11 ms | 1-3 ms | 1.0 ms |
| overhead, last 1,600 batches, median / mean | 0.271 / 0.273 | 0.21-0.23 | **0.208 / 0.204** |
| overhead, whole run, median / mean | 0.286 / 0.296 | 0.23-0.25 | **0.216 / 0.220** |
| overhead by window (median): 100-1000 / 1000-3000 / 3000-6143 | 0.31 / 0.30 / 0.27 | - | 0.250 / 0.222 / 0.209 |
| stream wall | 77.0 min | 64-68 min | **66.3 min** |
| end to end (pilot + stream + save) | 82 min | 69-73 min | **71.3 min** (2.5 + 66.3 + 2.5) |
| gate (v1 + loop vs v3 row split + stacked) | - | bitwise except sums | bitwise except sums (4.5e-9, the same as with R = 1) |
| redo batches / seeds short | 47 / 0 | 40-55 / 0 | 47 / 0 |
| nvidia-smi logger | "died" | dies within minutes | **alive the whole run** (5,846 lines; see below) |

Stream: 66.3 min = forward 48.8 + atlas 11.6 + paired timing forwards 2.0 + checkpoints 3.8 (host batch
building 0.0). **Forward plus atlas: 60.4 min; the README number.** The paired timing forwards and the
checkpoints are run configuration, not cost of the method.

Notes:
- Row split: 1.7x, not the 4-6x predicted; flat past R = 4. The kernel is no longer limited by the
  number of programs. My guess, unchecked: the int8 mask writes (49 MB per layer in 16-byte pieces at
  stride n). Not pursued (scope: two changes).
- The sums do not depend on R (fp32 within a 16-row block, fp64 across blocks, exact here), so the
  gate's sum difference against v1 is identical to the science pass's.
- nvidia-smi: the logger was alive at every ps (2 min interval) and at the end (1 h 24 min, 4,997 lines
  at the end of the pass). The three earlier "deaths" were a stale cache: a plain GET of the file's
  contents URL returned a frozen copy while the file kept growing; a unique-query GET returned it
  current. Fixed in `infra/lambda_api.py` (209afe4).
- The held-out atlas took 11 min, not 2-4: without seeds it warms up densely (961 layer redos over 130
  batches), and `--edges-from` loads the full 10.85 GB atlas for the edges.
- Atlas #2 vs #1: exact-field digests are recorded for #2; #1's are not yet computed (asia-south-1 had no
  capacity). Prediction: identical.

Data for the ruler (same job; no coherence computed):

| quantity | predicted | observed |
|---|---|---|
| held-out atlas | 4.08M tokens, 2-4 min | 4,080,811 tokens, 130 batches, 11 min (dense warm-up, see above) |
| top-64 lists, pole 0: median distinct proteins / UniRef50 clusters | clusters 10-25 | **63 / 32** |
| top-64 lists with < 10 clusters, pole 0 | 25-50% | **6.2%**: layer 0 94% (median 2 clusters), layer 1 24%, layers 2-32 1-5% |
| reservoir bins q99.999 / q99.99: median clusters; bins < 10 | 25-45, 3-15% / 45-60, 1-5% | **55, 0.0% / 61, 0.0%** |
| reservoir bins q99.9 and lower: bins < 10 clusters | < 1% | 0.0% (median 63-64 clusters of 64 samples) |
| AFDB descriptors for the whole build | computed | **not computed**: swissprot_pdb_v6.tar from EBI arrived at ~3.5 MB/s (62,466 of ~550k models in 32 min); the A100 was stopped |

The 2,000-protein sample (`scripts/afdb_descriptors.py`, from here): 3 sequence mismatches against
2026_03 (0.15%), 68 without an AFDB model (3.4%: 42 viral, 8 with non-standard residues, 18 other);
pLDDT < 70 for 16.3% of residues, < 50 for 10.2%, 50-70 for 6.1%. UniRef50 mapping of the build
(UniProt ID mapping): 574,463 of 575,748 accessions mapped, 193,718 clusters
(`lists/swissprot_2026_03_uniref50.tsv.gz`).

## 2026-11-04 (night) - AlphaFold DB descriptors for the whole build; checksums of committed results

`descriptors_20261104/`: whole-build summary, job log, and a sha256 manifest of every run file on the
us-east-1 filesystem. The descriptor file (`descriptors_swissprot_2026_03.pt`: ss3, contacts, burial,
pLDDT, phi/psi, C-beta coordinates, per residue) is on `esm-atlas-results` (us-east-1), next to atlas
copy #2. Instance: gpu_1x_a10, us-east-1, 21:31-22:15 UTC.

| quantity | predicted | observed |
|---|---|---|
| archive (swissprot_pdb_v6.tar) | - | 28.58 GB; 24 ranged connections, 26 MB/s average (one connection: ~3.5 MB/s) |
| F1 models | - | 550,122, extracted in 36 s |
| accessions with descriptors | 95-97% | 95.47% (549,655 of 575,748); 194.8M residues; 21 min on 28 processes |
| sequence mismatch vs 2026_03 | 0.1-0.3% | **0.074%** (426) |
| no AFDB model | 3-4% | 4.46% (25,667) |
| residues pLDDT < 70 / < 50 / 50-70 / >= 90 | 15-18% / 9-12% | 16.06% / 10.07% / 6.00% / 64.6% |
| held-out set: entries with a model; residues < 70 / < 50 | - | **7,208 of 10,010 (72%)**; 22.3% / 14.4% (more disordered than the rest: 16.0% / 10.0%) |
| 3-state among pLDDT >= 70 (loop / helix / strand) | - | 35.1% / 45.4% / 19.6% |

Checksums: every committed copy of a us-east-1 run file matches its filesystem original byte for byte
(phase 1, A10 profile, follow-up), except the follow-up's `ps.log` and `smi.csv`, which were snapshots
taken while the instance was still running (both kept growing for 35 more minutes). Runs on
asia-south-1 (science pass) and us-west-1 (impl_ab) are not yet checked.

## 2026-11-05 - Measurement #2: synthetic columns through the atlas and the ruler (CPU, this container)

`synthetic_ruler_20261105/`: per draw, the column table, the synthetic atlas (Atlas code as for model units,
K = K_r = 64, six edges to q99.999), blind calls per rule and the unblinded gate (`scripts/eval_synthetic.py`).
Proteins: the 60,000 non-held-out entries of the descriptor subset exported by job `atlas-wab-20261105-0140`
(67,208 proteins; on `esm-atlas-results` under `runs/atlas-wab-20261105-0140/desc_subset`), cropped to 1,022
residues as the model streams are: 20,557,222 residues, 162 columns (9 types x alpha {0, 0.5, 1, 2, 3, 4} x 3).
Generator ~90 s, ruler ~30 s per rule. Predictions: `measurements_1_2_predictions.md` (#2, #2b, #2c).

**Defects found before reading anything.** (1) 1,705 of the 60,000 proteins are longer than 1,024 residues;
token ids pack (protein, position) with stride 1,024, so their positions >= 1,024 (658,620 tokens, 3.1%)
aliased onto the next protein's ids and the ruler mapped them to wrong rows. Real runs crop at 1,022 and never
produce these ids. Fixed by cropping in `load_subset` and a stride/length-checked row map in the ruler; the
uncropped calls are kept (`design_uncropped_defect/`): the gate moved by at most one column per type. (2) The
gate script counted alpha = 0.5 columns as noise; alpha = 0 and noise are now one bucket (42 columns, not the
72 my prediction table said) and alpha = 0.5 is reported separately as "any association".

**Gate (alpha >= 1, 12 columns per type; noise = the 42 pure-noise columns).** Pre-registered cells in bold.

| column | predicted (v1) | v1 design | v1 draw 1 | v1 draw 2 | predicted (v1.1) | v1.1 draw 1 | v1.1 draw 2 | predicted (v1.2) | v1.2 draw 2 | v1.2 design / draw 1 (in-sample) |
|---|---|---|---|---|---|---|---|---|---|---|
| helix | 12 | **12** | 12 | 12 | 12 | **12** | 12 | 12 | **12** | 12 / 12 |
| buried | 9-12 | **12** | 12 | 12 | 10-12 | **1** | 0 | 9-12 | **12** | 12 / 12 |
| trp | 12 | **9** | 8 | 9 | 12 | **11** | 12 | 11-12 | **12** | 12 / 11 |
| lys_helix | 12 | **9** | 9 | 10 | 12 | **12** | 12 | 12 | **12** | 12 / 12 |
| two_job | 10-12 | **9** | 7 | 7 | 11-12 | **12** | 12 | 11-12 | **12** | 12 / 12 |
| ordered | 10-12 | **5** | 8 | 4 | 11-12 | **12** | 12 | 11-12 | **12** | 12 / 12 |
| narrow ("peak only") | 6-9 | **8** | 10 | 9 | 7-10 | **10** | 9 | 7-10 | **9** | 8 / 10 |
| hydro_win | 12 | **12** | 11 | 12 | 12 | **12** | 12 | 12 | **12** | 12 / 11 |
| noise (of 42) | ">= 69/72" | **40** | 42 | 39 | >= 41 | **42** | 42 | >= 41 | **42** | 42 / 42 |

narrow was associated in 12/12 under every rule and draw. hydro_win, a sequence-only column (mean hydropathy of
the +-3 window), is called structural under v1 (primary state5 or rama4, persistent to q99) and context under
v1.1/v1.2: as predicted, the ruler cannot tell a window-composition column from a structural one; that needs
the +-3 window baseline, not a better null.

**v1 failed its gate** on trp, lys_helix, two_job and ordered, and the failures repeat on two fresh draws, so
they are the rule, not the draw:
- composition: "residue-specific" required an effective number of residues <= 4, which measures dominance, not
  enrichment: Trp at alpha = 1 is 33-47% of the peak (25-36x its 1.3% background) and was "no association";
  lys_helix at alpha = 1 (K 59-63%, effective number 4.8-5.1) was "structural" instead of "context";
- several modes: a fixed log2 >= 0.5 per state on the marginal penalises mixtures (a sample split between two
  modes dilutes each); the secondary mode sat at log2 0.13-0.44 in the misses;
- false positives: the primary descriptor's p is the smallest of three and went into BH uncorrected; 5 of 126
  pure-noise columns across the three draws were called structural (0 of 126 under v1.1/v1.2).

**v1.1** (each change traced to one failure; pre-registered on draw 1): primary p x 3; residues tested against
P(residue | state on the primary descriptor) with residue modes (p < 0.01, log2 >= 1); second mode tested
among the samples outside the top mode. It fixed all four and **broke buried (1/12, predicted 10-12)**:
11 buried columns became "context" with residue modes G, A, S, P. The burial descriptor is a C-beta count
within 10 A (C-alpha for Gly); its extreme is Gly/Ala-rich, and the tercile does not absorb that. I named the
risk and underestimated it by a factor of ten.
**v1.2** (pre-registered on draw 2): residues conditioned on the descriptor's finest value (the raw count) and
tested with the largest standardised residue excess instead of the 19-df G (lys_helix at alpha 0.5 had a K mode
in 3/3 but passed the G in 1/3). Draw 2: every gate cell inside its prediction.

**Null calibration** (pure-noise columns): v1 descriptor tests, 630 per draw at all five levels: p <= 0.01 in
1.3% / 1.6% / 0.3%, p <= 0.05 in 4.8% / 4.4% / 5.2% (design uncropped / design / draw 1). v1.2's residue max
statistic, 378 tests over three draws: 0% at <= 0.01, 2.6% at <= 0.05, 43% at <= 0.5 (conservative: small
counts are lumpy). Conditional second-mode test: only 23 null cases (it runs only after a top mode); none <= 0.05.

**alpha = 0.5 (descriptive, "any association", 3 per type), v1.2 draw 2 vs prediction:** trp 3/3 (1-2),
lys_helix 1/3 (1-3), narrow 2/3 (0-1), hydro_win 2/3 (1-3), helix, buried, ordered, two_job 3/3 (3/3). Under v1,
trp at 0.5 was missed "by design" in all three draws; that design was the effective-number rule.

**Unexplained.** One trp column (draw 1, alpha 2, 97% W) is called context with a state5 mode under v1.1 and
v1.2: a W-only sample should match P(state | W). One in 36 trp/alpha >= 1 columns across draws; consistent with
chance at the per-column rate BH allows, but not shown to be chance.

**What this does and does not license.** v1.2 passed a pre-registered gate on a fresh noise draw. The fresh draw
protects against fitting the noise, not against fitting the nine column types: two of v1.2's three changes were
written after seeing how these types fail. The next check is column types the rule was not designed on (ideally
written by someone other than me), then the decision-rule sign-off; no real column has been scored.

## 2026-11-05 - Measurement #3: what a unit writes and what silencing it does (8M, CPU)

`unit_writes_8M_20261105/` (`scripts/unit_writes.py`; predictions `measurement_3_predictions.md`): ESM-2 8M
fp32, 400 random non-held-out proteins (133,506 residues), 50 random units per layer, three tokens per unit
(largest h in the sample, the token nearest the median, smallest h): 900 cases, 1,800 silenced forwards,
201 s on one thread. Re-running a forward reproduces the log-probs exactly (noise floor 0.0), so effects
down to 1e-12 are real. Medians over the 50 units of each layer:

| layer | GELU(h): pos / median / neg | written / residual: pos / median / neg | KL at the position: pos / median / neg | pos KL / baseline KL | units with \|dlog p(true)\| > 0.1 at pos |
|---|---|---|---|---|---|
| 0 | 1.08 / -0.155 / -0.0027 | 0.24 / 0.032 / 5.4e-4 | 9.3e-5 / 8.4e-7 / 2.5e-10 | 143 | 2% |
| 1 | 1.66 / -0.136 / -0.0016 | 0.18 / 0.014 / 1.5e-4 | 2.9e-4 / 5.6e-7 / 2.6e-10 | 571 | 4% |
| 2 | 1.74 / -0.061 / -0.0091 | 0.14 / 0.009 / 6.0e-4 | 7.0e-4 / 4.0e-7 / 2.6e-8 | 601 | 10% |
| 3 | 2.96 / -0.140 / -6.7e-5 | 0.15 / 0.009 / 3.5e-6 | 2.0e-3 / 1.9e-6 / 1.4e-12 | 1,900 | 14% |
| 4 | 4.31 / -0.161 / 0 | 0.20 / 0.008 / 0 | 3.0e-3 / 2.4e-6 / 0 | 2,060 | 18% |
| 5 | 4.93 / -0.155 / 0 | 0.24 / 0.007 / 0 | 6.5e-3 / 5.6e-6 / 0 | 1,840 | 20% |

Baseline = a random other unit of the same layer silenced at the same position (median KL 5e-7 to 4e-6).

Against the predictions:
- **Negative pole is the off state at 8M, across 300 units:** median |GELU| at the negative extreme
  <= 0.009 in every layer, exactly 0 in layers 4-5; silencing there moves the logits by a median KL <= 3e-8,
  below the median token's in every layer. Neither contradiction criterion was met.
- Inside the predicted ranges: written/residual at the positive extreme (0.14-0.24), median-token KL,
  every negative-extreme cell, |dlog p(true)| at the positive extreme (0.005-0.029), median-token KL ratio
  to baseline (0.7-3.0). Pooled over layers, 11% of units move log p(true) by > 0.1 at their positive extreme
  (predicted >= 10%), 0% at the negative extreme (predicted < 2%).
- Outside: GELU at the positive extreme in layers 0-2 is 1.1-1.7, under the predicted 2-8 (400 proteins
  are 0.07% of the stream; the atlas extremes are larger); written/residual at the median token in layers
  0-1 (0.032, 0.014) above the predicted 0.01; unit/baseline KL at the positive extreme in layers 3-5 above the
  predicted 1,000x.
- **Travel, wrong:** predicted exactly 0 for layer 5 (it is: nothing after the last FFN mixes positions) and,
  for layers 0-4, a largest KL elsewhere below the KL at the position. Observed median travel above the
  position's own KL in layers 1-3 (3.6e-4 vs 2.9e-4, 1.6e-3 vs 7.0e-4, 9.5e-3 vs 2.0e-3), up to 3.2 nats at
  another position. Part of this is a maximum over hundreds of positions set against one value. The rest
  is concentrated: in the 10 of 250 cases where the write exceeds the residual it is added to (ratio 1.0-3.1),
  every one moves another position more than its own (median 0.36 vs 0.020); below that ratio, 124 of 240.
  Silencing such a write replaces that position's representation, which later layers read elsewhere
  through attention. For interventions, "what the unit does" is not confined to the position where it
  fires.

## 2026-11-05 - Measurement #1: the complete workload done three ways (A100 40 GB, us-east-1)

`workload_ab_20261105/` (job `jobs/workload_ab`, 01:44-02:45 UTC; predictions `measurements_1_2_predictions.md`
#1). Same plan, same batches, every batch paired with a hook-free forward (overhead = atlas pass / forward - 1).
naive = our dense path on every layer of every batch (`--naive`: exact, the correctness reference);
baukit = baukit's `runningstats.TopK` algorithm re-implemented (installing the package was refused here),
top-K only. Prefix = batches 0-59 from fresh state; steady = batches 4500-4559 (filtered resumed from the
follow-up's checkpoint, the others fresh).

**Outputs.** Complete filtered vs complete naive: bitwise equal on every field (top-K values and ids,
reservoir tokens, values and keys, bin masses, enrichment counts, sums, Gram; n_seen 1,876,042). Top-K
only, filtered vs naive: equal. baukit vs ours: values equal (max difference 0.0); ids differ at
12,090,302 of 21,626,880 positions, predicted "hundreds to a few thousand at ties". Cause, shown at 8M on CPU
(`scripts/topk_baseline.py` now sorts both lists canonically): in bf16, 71% of top-64 slots sit in exact-value
ties and torch.topk orders ties arbitrarily; after a canonical sort every remaining difference (2.0% of slots)
is in the tie group at the K-th value, none above it; fp32 has 0. The 650M audit runs in the next job.

**Overhead** (mean / median over 60 batches):

| run | predicted | prefix (0-59) | steady (4500-4559) |
|---|---|---|---|
| complete, filtered | 0.28-0.38 / 0.20-0.22 | 12.8 / 3.15 (see below) | **0.207 / 0.208** |
| complete, naive | 8-20 | 27.4 / 27.7 | 27.1 / 27.6 |
| top-K only, filtered | 0.05-0.12 | 0.221 / 0.081 (0.077 / 0.070 on 30-59) | - |
| top-K only, naive | 0.3-0.6 | 1.53 / 1.53 | 1.49 / 1.51 |
| top-K only, baukit's algorithm | 0.3-0.5 | 0.903 / 0.918 | 0.895 / 0.919 |

Forward 0.465-0.475 s per batch throughout.

**The filtered prefix measured the wrong thing, by my job design.** `--max-batches 60` makes a complete
60-batch run, and thresholds are seeded for the stream being run (1.9M tokens, not 194M): reservoir bins
whose whole-stream count is near K_r stay "underfilled" and admit everything above their lower edge, the
candidate buffer overflows, and the layer is redone densely. 22 of 60 batches redid all 33 layers and 36 redid
some; the 24 without a redo had 0.632 / 0.482. A real run's first 60 batches are seeded for the full stream:
the follow-up's gate run on the same batches (same code path, `--stop-after`) had 0.765 / 0.298 with 12 redo
batches (layer 0), which is what the prediction was built on. Top-K-only seeds do not depend on stream length,
so that prefix row stands.

**What it shows.**
- At steady state the complete collection (top-K, reservoirs, subsample statistics, Gram) costs 0.21 of a
  forward; baukit's top-K alone costs 0.92, our dense top-K alone 1.51. Doing everything with the filter is
  4.4x cheaper than an unfiltered top-K alone.
- Complete naive / complete filtered = 133x (median; predicted 40-80x). This is the filter against our own
  dense path, which is a sync-heavy correctness reference, not a tuned dense implementation, so it bounds the
  gain from filtering from above. The fair external comparison exists only for top-K: filtered 0.07-0.08 vs
  baukit 0.92 (12x) vs naive 1.5 (19x); naive and baukit within 2x of each other (1.66x), as predicted.
- Not separated: how much of the 12x is fewer candidates and how much is the fused Triton compaction; both
  are part of the filtered path. Not explained (no profile was taken): why our dense path costs 27x a forward
  for the complete workload (expected 8-20x); per-element key, hash and bin work over 11G elements per batch
  is the likely cost, unmeasured.

## 2026-11-05 - Measurement #7, part 2: aligning the ubiquitin deep mutational scan (CPU)

`dms_alignment_20261105/` (`scripts/dms_align.py`; predictions `measurement_7_predictions.md`). Source: the
per-condition fitness CSVs of Mavor et al. 2016 (eLife 15802) in github.com/fraser-lab/pubs2014 (commit in
`source_commit.txt`, sha256 of each CSV in the JSON; the CSVs are not copied here, the repository has no
licence; its pickles were not loaded).

| quantity | predicted | observed |
|---|---|---|
| scan wild type vs human | yeast; differs at 19, 24, 28 | yeast; P19S, E24D, A28S, nothing else |
| rows; missense scored (DMSO) | 1,596; >= 90% of 1,444 | 1,596; 1,298 (89.9%); other conditions 1,218-1,256 |
| humanising substitutions S19P / D24E / S28A (DMSO) | > -0.1 (stop ~ -0.5) | +0.021 / -0.012 / -0.006 |
| allele fitness, Pearson with DMSO | 0.6-0.9 | caffeine 0.84, DTT 0.81, HU 0.87, MG132 0.91 |
| positions whose condition means differ by > 0.1 | 5-20 of 76 | 45 (raw ranges; replicate SDs are 0.08-0.27 per allele) |

Corrected for noise and for condition-wide effects (each condition mapped onto DMSO by an affine fit over all
alleles: MG132 slope 0.81, DTT offset -0.078; then Cochran's Q over the five calibrated position means, SE from
the replicate SDs of each position's alleles), 43 of 70 testable positions still differ between conditions at
Bonferroni p < 0.01, and 7 have no detected condition dependence at p > 0.1 (8, 13, 33, 42, 44, 54, 67; not
"stable": that needs an equivalence test, see the revised ubiquitin section). That rests on three-replicate SDs
being honest errors; if they understate the noise by 2x, Q is 4x too large. Either way the prediction (5-20)
was too low.

What it decides: a unit-level check against this scan compares human-sequence activations with yeast
fitness (the three differing positions are themselves near-neutral in yeast), and most positions' tolerance
moves with the condition. A comparison should name the condition (DMSO primary) or use positions shown equivalent across
conditions at a stated margin; "the DMS says this position matters" is not one statement here.

## 2026-11-05 - Measurement #4: four relations between units (8M, CPU on the new 8M atlases)

`families_8M_20261105/` (`scripts/families.py`; predictions `measurement_4_predictions.md`). Atlases: job
`atlas-batch-20261105-0258` (A10, us-east-1): ESM-2 8M over the full non-held-out stream (194,177,125
residues), K 64, K_r 256, bf16, seeds verified (0 of 73,763 seeded bins short); controls: weights permuted
within every tensor, same stream, hashes and plan. Per layer, positive pole, all 818,560 unit pairs.
Co-activation = Pearson r of pre-GELU h on the hashed 5% subsample (Gram); input / output = cosine of
W_in rows / W_out columns; tail = Jaccard of the token sets above q99.9, estimated from the bottom-256
reservoir sketches (keys recomputed from token ids; checked equal to the stored keys).

| Spearman over pairs, by layer 0-5 | trained | shuffled (seed 2) | predicted |
|---|---|---|---|
| co-activation ~ input weights | 0.61 / 0.63 / 0.62 / 0.61 / 0.65 / 0.62 | 0.22 at every layer | 0.3-0.7, trained and control alike |
| co-activation ~ tail | 0.42 / 0.35 / 0.27 / 0.21 / 0.34 / 0.31 | 0.26-0.35 | 0.05-0.3 |
| input ~ tail | 0.31 / 0.25 / 0.20 / 0.23 / 0.29 / 0.28 | 0.10-0.13 | - |
| input ~ output, co-activation ~ output, output ~ tail | -0.01 to 0.04 | 0.00 | -0.1 to 0.2 |

Top-5 neighbour agreement (median over units): co-activation vs input 0.4 (layers 0-2), 0.2 (3-5);
co-activation vs tail and input vs tail 0.2 in layers 0-1, 0 later; anything vs output 0 (predicted
10-40%, 5-25%, < 5%). Near-duplicate tails (Jaccard >= 0.5): 3, 1, 2, 3, 8, 0 pairs per layer (predicted
0.01-0.5% of pairs, i.e. 80-4,000, in layers 0-1); 0 in the control.

What this says:
- Co-activation tracks input weights three times more strongly in the trained model (0.62) than in the
  control (0.22). I predicted the two would match, as any two linear read-outs of a shared input should; the
  difference means training aligned the input weights with the directions the residual stream varies along.
- What a unit writes is unrelated, by cosine, to what it reads, when it fires and which tokens drive it,
  in the trained model as in the control: a family defined by inputs or firing is not a family by output.
- Three concrete disagreements (trained, `pretrained/disagreements.jsonl`):
  1. same tokens, opposed writes: layer 1 units 86 and 1200, both valine units (top-64: V53 / V43),
     co-activation 0.91, input cosine 0.81, tail Jaccard 0.50, output cosine **-0.66**; layer 0 units 922 and
     1134, both all-Cys top-64 lists, tail 0.52, output -0.29;
  2. correlated bulk, disjoint extremes: layer 0 units 797 (top-64 all Asn) and 1191 (K59 R4): co-activation
     0.75, tail Jaccard 0; a co-activation family would merge an Asn unit with a Lys/Arg unit;
  3. near-identical input weights with different tails: none (no pair with input cosine >= 0.8 and tail
     <= 0.05); where input weights nearly coincide, tails overlap (0.5-0.6).
- Across the three control seeds, co-activation ~ input weights is 0.209-0.227 in every layer and seed, and
  near-duplicate tails number 0-1 per layer: the trained/control separation (0.61-0.65 vs ~0.22) is far
  outside the control spread (`shuffled_seed1/2/3`).

## 2026-11-05 - Job batch_20261105: 650M tie audit, 8M atlases (A10, us-east-1)

`batch_20261105/` (predictions `batch_20261105_predictions.md`, with an addendum written before launch: no
A100/H100 anywhere at 02:55, so parts A, C, D ran on an A10 and the K sweep (B) waits for an A100).

**A. Tie audit at 650M** (first 60 batches, filtered top-K vs baukit's algorithm, canonical comparison):

| quantity | predicted | observed |
|---|---|---|
| values equal | yes | yes (max difference 0.0) |
| ids differing above the K-th value | **0** | **0** |
| slots inside exact-value ties | 50-80% | 79.9% (17,273,097 of 21,626,880) |
| ids differing at the K-th value | 1-5% | 3.55% (768,485), all in the boundary tie group |
| columns whose K-th value ties with a token outside the list | 50-80% | 76.4% (258,121 of 337,920) |

Both are exact top-K lists; they differ only in which tokens of the last tie group get in, which our key
fixes (smaller token id) and torch.topk does not. The positional count (12,090,302) is identical to the
A100 run's. At bf16, 80% of a 650M unit's top-64 sits in exact ties: any statement about "the" top-64 tokens
of a unit depends on a tie rule.

**C. 8M atlases** (K 64, K_r 256, bf16): pretrained full stream 194,177,125 residues in 494 s (8.2 min;
predicted 10-20 min on the A10), 0 redo batches, paired overhead mean 0.674 / median 0.357 (predicted 1-4:
the atlas is cheaper relative to the 8M forward than I assumed); seeds verified (0 of 73,763 short);
held-out (same edges) 4,080,811 residues in 216 s; atlas.pt 0.61 GB, slim copies 0.214 GB. Controls: **two
of three failed** with CUDA out of memory, because three pilots ran concurrently (6-9 GB each on a 22 GB A10;
the concurrency was sized for a 40 GB A100 and the CPU smoke ran them one at a time, so it could not catch
this). Seed 2 completed (597 s). The launcher was stopped before the job ended, seeds 1 and 3 were rerun one
at a time on the same instance as a second job (part E, smoke-tested on CPU first), and supervision was
re-attached with the original cap; part C now runs controls concurrently only on >= 40 GB GPUs.

**D. #7, part 1: ubiquitin copies in unit samples.** Copies of human ubiquitin (>= 70/76 identical, ungapped):
394 in 146 proteins (155 exact in 50; 5 proteins held out), 29,260 residues in the non-held-out stream
(0.0151%; `ubiquitin_copies.tsv`).

| quantity | 650M (337,920 columns) | 8M trained (15,360) | 8M shuffled, seeds 1 / 2 / 3 | predicted (650M / 8M / shuffled) |
|---|---|---|---|---|
| columns with any copy residue in top-64 | 201 (0.06%) | 40 (0.26%) | 0 / 0 / 0 | 2-6% / 2-6% / 2-8% |
| columns with >= half of top-64 in copies | 39 | 4 | 0 / 0 / 0 | 0.1-1% / 0.1-1% / >= trained |
| columns whose top-64 are all copies | 1 (layer 15) | 0 | 0 | - |
| peak reservoir bins >= half copies | 9 | 1 | 0 / 0 / 0 | 0-50 / 0-10 / 0-10 |

Wrong by one to two orders of magnitude, and in the control column wrong in direction: copies dominate a few
dozen 650M units (spread over layers 3-32), not hundreds to thousands, and no shuffled-weight control puts a
copy residue in any top-64 at all. Independent draws would put one in ~1% of lists; trained top-64 lists sit
below that and control lists far below (their tops are residue identity spread over many proteins, see #6).
So ubiquitin redundancy matters for those 39 units, where a ubiquitin-based check is a check on copies of one
sequence, and is negligible elsewhere; where it shows up, the trained model put it there.

## 2026-11-05 - Measurement #6: three shuffled-weight controls at 8M

`controls_8M_20261105/` (`scripts/controls_summary.py`; predictions `measurement_6_predictions.md`).
Medians over the 1,280 units of each layer, positive pole; trained vs the three controls (seeds 1, 2, 3):

| layer | effective residues, top-64 | effective residues, peak sample | top-64 from <= 3 proteins | top-64 one residue |
|---|---|---|---|---|
| 0 | 2.94 vs 1.26 / 1.31 / 1.37 | 5.45 vs 1.60 / 1.71 / 1.76 | 2.8% vs 7.7 / 8.1 / 7.8% | 22% vs 57 / 57 / 54% |
| 1 | 2.94 vs 1.75 / 1.80 / 1.78 | 6.44 vs 2.56 / 2.49 / 2.69 | 0.5% vs 9.1 / 8.1 / 10.4% | 23% vs 42 / 40 / 41% |
| 2 | 3.72 vs 1.85 / 1.90 / 1.98 | 8.06 vs 2.70 / 2.75 / 3.10 | 0% vs 11.4 / 8.6 / 9.4% | 17% vs 37 / 38 / 35% |
| 3 | 3.29 vs 1.89 / 1.92 / 1.95 | 7.86 vs 2.94 / 2.99 / 3.34 | 0% vs 13.6 / 7.8 / 13.3% | 18% vs 37 / 35 / 36% |
| 4 | 4.52 vs 2.13 / 2.07 / 2.22 | 10.49 vs 3.35 / 3.59 / 3.97 | 0% vs 13.4 / 8.0 / 12.0% | 12% vs 30 / 32 / 30% |
| 5 | 4.71 vs 2.27 / 2.37 / 2.37 | 10.51 vs 3.91 / 4.14 / 4.51 | 0.2% vs 14.3 / 9.4 / 13.7% | 11% vs 31 / 25 / 27% |

- Direction as predicted: untrained units are more residue-specific than trained ones at every layer
  (effective residues in the top-64 1.3-2.4 vs 2.9-4.7). "This unit fires on one residue" is the easiest
  result to obtain from a network that learned nothing.
- Spread across control seeds, as a fraction of the trained-minus-control difference: 2-5% for both residue
  measures (predicted < 10%), 5-7% for the one-residue share except layer 5 (18%), and 4-28% for the
  protein-concentration share, whose trained value is ~0. One control seed separates trained from untrained
  on residue diversity; on protein concentration it would not be enough beyond layer 0.
- Outside the predictions: the trained peak samples are more diverse than predicted in layers 2-5 (7.9-10.5
  vs 2-8), and both trained and control top-64 lists are less protein-concentrated than predicted (trained
  0-2.8% vs 5-30%; controls 8-14% vs 30-70%).

## 2026-11-05 - Measurement #8: K swept over 16 / 64 / 256 (650M, A100 40 GB, us-east-1)

`k_sweep_650m_20261105/` (job `atlas-ksweep-20261105-0358`, part B of `jobs/batch_20261105`, 04:00-04:23 UTC;
predictions `batch_20261105_predictions.md` B). Complete workload, first 200 batches, thresholds seeded for
the full stream (`--stop-after`), K_r 64, every batch paired. Batches 100-199 without a redo (94 per K):

| K | overhead, batches 100-199, median (predicted) | mean / median, all 200 | top-K candidates per batch (predicted) | atlas excess, median | redo batches (predicted) |
|---|---|---|---|---|---|
| 16 | 0.252 (0.24-0.27) | 0.499 / 0.256 | 39k (~40k) | 117.5 ms | 18 (15-30) |
| 64 | 0.260 (0.25-0.28) | 0.461 / 0.267 | 146k (~150k) | 121.7 ms | 20 (15-30) |
| 256 | 0.323 (0.28-0.40) | 0.592 / 0.341 | 555k (~600k) | 150.2 ms | 20 (15-30) |

Reservoir candidates are the same at every K (228k per batch), as they should be. Marginal cost per top-K
candidate: 38.7 ms per million from K 16 to 64, **69.9 ms per million from 64 to 256** (predicted 15-60):
the cost per candidate grows with K, as a merge whose sort width is K would make it. At K = 64 the top-K
candidates account for ~6-10 ms of a 122 ms excess in these early batches; the rest is per-token work that
does not depend on K. K = 256 for every bin would cost +24% atlas time early in the stream; K_r = 256 at
the lower bins (agreed for the next build) changes reservoir candidates, which this sweep did not vary.

## 2026-11-05 - Ruler v1.2 frozen and challenged (CPU)

`ruler_challenges_20261105/` (sets A-D; `scripts/ruler_challenges.py`; expectations `ruler_challenges_predictions.md`,
committed before any challenge column existed). Rule: `atlas/ruler.py` at aab0c82, v1.2, unchanged throughout.
60,000 proteins (20.56M residues), 414 columns per set (240 null), each streamed through the Atlas, scored, called.

**Whole-procedure error rates** (selection, reservoir sampling, de-duplication, scoring, BH, calls), Clopper-Pearson 95%:

| set | null columns called (as built, 42% signal) | false discovery proportion | 80%-signal mix | all-null mix |
|---|---|---|---|---|
| A fp32 | 0 / 240 (0-1.5%) | 0 / 148 (0-2.5%) | 1 / 43 (0.06-12.3%) | 0 / 240 |
| B bf16 | 1 / 240 (0.01-2.3%) | 1 / 149 | 1 / 43 | 0 / 240 |
| C ids permuted | 0 / 240 | 0 / 147 | 0 / 43 (0-8.2%) | 0 / 240 |
| D 5% duplicated | 1 / 240 (0.01-2.3%) | 1 / 149 (0.02-3.7%) | 1 / 43 | 0 / 240 |

The sets share their noise, so they are not independent replicates. The 80% mix has only 43 null columns,
and its interval reaches 12%; bounding the error rate where most columns are real needs more null columns
than this set has.

**Challenges** (set A; alpha >= 1, 12 columns per type unless stated):

| challenge | expected | observed |
|---|---|---|
| motif N-x-[ST] (sequence only) | composition (N) >= 9; structural <= 6 | composition, residue mode N, **12 / 12**; structural 0 |
| motif G-x-G-x-x-G (sequence only) | structural / context >= 8, mode G | **5 / 12** context (alpha 3-4); composition G 4; none 3 (alpha 1) |
| exposed strand (conjunction) | associated 12; primary state5 6-12; burial3 also significant in the scores >= 10 | associated 12 (structural 9, context 3); primary state5 12 / 12; burial3 p <= 0.01 in the stored scores 12 / 12; the call names only state5 |
| balanced interaction (marginals planted equal to background; checked: within 0.0001) | "no association" >= 8 | **12 / 12 not detected** |
| correlated noise, null columns (30 each) | <= 2 / 30 each | protein effect 0 / 30, cluster effect 0 / 30, smooth 0 / 30 (2 insufficient samples); pooled 0 / 90 (0-4.0%) |
| correlated noise, clusters at the peak | protein / cluster 20-50, smooth 45-62 | 61 / 58 / **33.5** (smooth noise concentrates the peak in fewer proteins; protein and cluster effects barely do) |
| correlated noise, power (helix, trp, ordered) | alpha 3 >= 8 / 9; alpha 1 up to half lost | 9 / 9 at both alphas in every model |
| bf16 (B) vs fp32 | class agrees >= 90% | 161 / 162 (149 with the same primary and mode label) |
| ids permuted (C) vs A | >= 85% | 152 / 162; every disagreement at the margin (structural <-> context as a residue mode crosses threshold; composition <-> none at alpha 1) |
| 5% duplicated (D) vs A | >= 85% | 156 / 162; median clusters at the peak 64 -> 62 (de-duplication absorbs the copies) |

**Window baseline v0 (declared diagnostic, not part of the call) failed.** It said "structure beyond the +-5
window" for every helix, buried, ordered, two_job, narrow, lys_helix and exposed-strand call (correct: those
columns are structural), and also for all 5 G-x-G-x-x-G calls and 4 of 12 hydro_win calls, which are
sequence-only: 9 false claims out of 17 sequence-only structural / context calls. For the motif, the window
reweighting collapses onto a handful of G-rich windows (effective sample size median 21, against 300-50,000
elsewhere); for hydro_win (ESS 328) an independent-sites model approximates a window mean too loosely.

**Verdict against the criteria fixed beforehand.**
- *Wording (round 2):* the challenges **found no failure of v1.2 under the tested conditions**. That is not
  independent validation. The sets share their noise, and the error rate where most columns carry signal is
  bounded only loosely (upper 95% bound 12.3% at 80% signal, 43 null columns).
- *No necessary correction to v1.2.* Error control held in every set, noise model and transformation (no
  95% interval lies above 2%), and no call asserted something false in its own terms: N and G composition
  calls are true, and the exposed-strand association is real.
- *Necessary correction to the baseline before any "structure beyond sequence" statement.* v0 must not be
  used; a sequence model that can represent residue conjunctions (and a stated ESS floor) is needed, then
  re-challenged on the same sequence-only columns.
- *Documented limitations, carried into every real-column report:*
  1. a conjunction is reported through one descriptor, and the other is visible only in the scores;
  2. a balanced interaction is "not detected by this ruler" (12 / 12), which is not the same as "no
     coherent computation";
  3. sequence-only columns can be called structural or context: 12 / 12 hydro_win, 5 / 12 motif;
  4. calls move with bf16 or relabelling only at the detection margin;
  5. the error rate at high signal fractions is bounded only loosely (upper bound 12%).

## 2026-11-05 - Valine pair: layer 1 units 86 and 1200 of ESM-2 8M, on fresh proteins

`valine_pair_20261105/` (`scripts/pair_silencing.py`; predictions `mechanistic_round1_predictions.md` section 1,
with an amendment before the real run: the main group is "both above q99.9", because on fresh proteins unit
86's q99 is below zero). 400 date-held-out proteins (164,947 residues); 44 positions where both fire; at each,
post-GELU silencing of 86, of 1200, of both, and halving both, against a control pair matched per position on
write magnitude.

From the weights: input cosine 0.81, output cosine -0.661, the most opposed write to 86's among all 1,280 units
of the layer; |w_out| 1.68 / 1.12.

| quantity ("both" positions, median) | pair | matched controls | predicted |
|---|---|---|---|
| Pearson r of the two units over all fresh tokens | 0.899 | - | 0.80-0.95 |
| top-64 that are V, each unit | **36%** | - | >= 60% |
| cosine of the residue-prediction changes, silence 86 vs silence 1200 | **-0.78** | +0.11 | -0.3 to -0.8 |
| joint / (individual + individual), prediction change | **0.40** | 0.80 | 0.3-0.7; controls 0.8-1.0 |
| Spearman of layer-2 changes, silence 86 vs silence 1200 | **-0.55** | +0.02 | -0.3 to -0.9 |
| halve both / silence both | 0.52 | - | 0.4-0.6 |
| KL(silence both) < KL(silence 86) | 45% of positions | - | >= 50% |
| masked prediction elsewhere, mean \|D log p(true)\| | 0.0052 / 0.0063 / 0.0057 (86 / 1200 / both) | 0.0023 / 0.0023 / 0.0046 | - |

**What their opposed writes do:** they act on a polar-versus-hydrophobic axis, not on valine. At the position,
silencing 86 lowers log p of E, Q and G by ~0.06 each and slightly raises I; silencing 1200 raises E, K and Q by
0.06-0.07 and lowers I and F. My prediction (V down, I / L / A up) was wrong. The same later units move in
opposite directions: layer-2 units 786, 785, 1142, 203 and 1235 are among the ten most changed by silencing 86
at most positions (786 at 36 of 44).

**Where the case is weak:**
- The pair never fires alone on fresh proteins. With one unit above q99.9 and the other below q90 there are
  0 positions of either kind, so the regime where their *difference* would matter cannot be isolated by
  selection. That needs graded interventions (e.g. scale one unit across its range while holding the other)
  rather than found positions.
- Unit 1200's write (2.35) is larger than any other unit's at those positions, so its "matched" control
  writes half as much (1.11). The controls show what a pair of unrelated units does, not one of equal size.
- At a valine the unmasked prediction is near-certain, so the KL there is ~1e-4. The masked readout at other
  positions, where the pair moves predictions 2.5x more than its controls, is the more informative.

Supported: units with similar inputs and opposed writes have opposed, partly cancelling effects on fresh
proteins (0.40 vs 0.80 sub-additivity; -0.78 vs +0.11). Not shown: what the cancellation accomplishes. The
next step is to vary each unit's activation independently and measure the polar / hydrophobic readout.

## 2026-11-05 - Ubiquitin, revised (CPU)

`ubiquitin_v2_20261105/` (`scripts/dms_align.py`, `scripts/ubq_dms_scores.py`; predictions
`mechanistic_round1_predictions.md` section 3). DMSO is the primary condition; the others are shown
separately, never pooled.

| ESM-2 masked-marginal score vs fitness (yeast wild type, Spearman) | DMSO (primary) | caffeine | DTT | HU | MG132 | predicted (DMSO) |
|---|---|---|---|---|---|---|
| 8M | **0.112** | 0.163 | 0.116 | 0.106 | 0.143 | 0.15-0.35 |
| 650M | **0.541** | 0.600 | 0.540 | 0.546 | 0.528 | 0.35-0.55 |

(n = 1,218-1,298 scored missense alleles per condition.) Yeast ubiquitin is itself in ESM-2's training data,
so this is an alignment check, not an independent test.

Conditions:
- Positions 8, 13, 33, 42, 44, 54 and 67 (p > 0.1 in the heterogeneity test) are positions with **no detected
  condition dependence**, not "stable".
- Equivalence (two one-sided tests, 90% intervals of each condition's calibrated deviation within the margin):
  at **+-0.1** (stop ~ -0.5, wild type ~ 0), **38 of 70** positions are equivalent across conditions,
  including all seven above. At **+-0.05**, only **4** are (33, 54, 64, 65). Predicted 0-7 at +-0.1: wrong.
- Many positions with detectable condition dependence are also equivalent at +-0.1: their differences are
  real but small. So "43 of 70 positions depend on condition" (part 2 above) overstates the size; the margin
  decides the answer, and it is a choice.

## 2026-11-05 - Travel, re-measured (8M, CPU)

`travel_v2_8M_20261105/` (`scripts/travel_v2.py`; predictions `mechanistic_round1_predictions.md` section 2).
The 300 positive-extreme cases of measurement #3: KL at every position (not only the largest elsewhere), by
distance; a control unit at the same position matched on write magnitude (achieved ratio 1.02-1.51 by layer);
and a masked task (15% of the other positions masked).

| layer | KL at i | median KL elsewhere / at i | sum elsewhere / at i | median near (\|d\| <= 3) / far (> 16) | unit / matched control, summed travel | masked / unmasked change, same positions | masked change: unit / control |
|---|---|---|---|---|---|---|---|
| 0 | 9.3e-5 | 0.0008 | 4.3 | 200 | 1.42 | 3.2 | 1.20 |
| 1 | 2.9e-4 | 0.0031 | 6.5 | 136 | 3.21 | 3.2 | 1.62 |
| 2 | 6.9e-4 | 0.0033 | 15.4 | 124 | 1.41 | 3.5 | 1.26 |
| 3 | 2.0e-3 | 0.0009 | 29.9 | 257 | 1.01 | 3.4 | 1.23 |
| 4 | 3.0e-3 | 7e-6 | 0.91 | 1,742 | 1.42 | 3.7 | 0.76 |
| 5 | 6.5e-3 | 0 | 0 | - | - | - | 0 |

Against the predictions: the per-position ratio (1/1000-1/100 predicted) holds in layers 0-3; the summed
ratio is above the predicted 1-10x in layers 2-3 (15, 30); near/far >= 10 holds (124-1,742); unit /
control 1-3 holds except layer 1 (3.2); masked / unmasked 3-30 holds (3.2-3.7); layer 5 is exactly 0 in both
tasks.

What survives: measured per position, silencing a unit at its strongest token moves any one other position
by 0.1-0.3% of what it moves its own. The earlier "moves other positions more than its own" compared a
maximum over hundreds of positions with one value. *Summed* over the protein, though, the effect elsewhere is
4-30x the effect at the position in layers 0-3, concentrated within three residues. It mostly tracks write
magnitude rather than unit identity: a control unit writing as much at the same position travels about as far
(1.0-1.4x except layer 1), in the masked task too. For interventions: where a unit fires is not where most of
its effect lands, but that is a property of large writes in early layers, not evidence about this unit.

## 2026-11-05 - Measurement #5: strict held-out set (A10, us-east-1)

`strict_heldout_20261105/` (job `jobs/strict_heldout`; predictions `measurement_5_predictions.md`, with an
addendum written between the two attempts). Two attempts: the first (11:07-11:29) downloaded the archive and
stopped by design when `uniref50.tar` turned out to hold XML only; the second (11:35-12:17) streamed it to FASTA.

**Reference.** UniRef50 release 2021_04 (27 Oct 2021), from `uniref2021_04.tar.gz` (169,188,630,923 bytes; MD5
444f0a7062a65a988ba1d5949d1d6419 matched in both attempts; 1,044 MB/s over 16 ranges in the second). Members of
`uniref50.tar`: README, uniref50.release_note, uniref50.xml.gz (25.1 GB), uniref.xsd. Converted: **49,874,565**
entries -> 49,874,565 representative sequences (as the release note says), of which **40,179,930 (80.6%)** are
their cluster's seed (predicted 30-70%). FASTA kept on `esm-atlas-results/cache/uniref50_2021_04/` (9.0 GB).
**Targets:** representative sequences only. **Queries:** the 10,010 Swiss-Prot 2026_03 entries created after
2021-11-17, full length. **Search:** MMseqs2 13-45111 (Ubuntu 22.04 package), `search -s 5.7 -e 1e-3
--max-seqs 1000 --alignment-mode 3`, no coverage filter in the search; 5,456,806 hits in 639 s.

| criterion (no hit meeting it) | survivors | predicted | with descriptors (AFDB model) |
|---|---|---|---|
| >= 50% identity over >= 80% of the representative (UniRef50's own rule, applied to the representative) | **451** (4.5%) | 5-20% | **231** |
| >= 50% over >= 80% of the query | 559 | - | 289 |
| >= 40% over >= 50% of either | 301 | - | 122 |
| >= 30% over >= 50% of either | **287** (2.9%) | 1-5% | **111** |
| any hit at E <= 1e-3 | **284** (2.8%) | 0.5-3% | **109** |

Of the 10,010, 9,726 have some hit; 7,208 have descriptors (predicted share of survivors with descriptors
40-70%: 51% at the 50% criterion, 38-39% at the stricter ones; the most novel entries are the least likely to
have a model).

**What a negative result establishes.** No *representative* of a 2021_04 UniRef50 cluster within the
criterion. UniRef50 membership is defined against the cluster's seed; here 80.6% of representatives are seeds,
so for about one cluster in five the sequence that defines membership was not searched. Members are only
>= 50% identical to their seed, and ESM-2 sampled UniRef90 members inside clusters, so "no hit >= 50% to any
representative" does not exclude a >= 50%-identical training member. The 30% / no-hit sets (287 / 284,
109-111 with descriptors) are the defensible "strict" held-out sets; even they are a statement about
sequence search at this sensitivity (-s 5.7), not proof that nothing related was in training. Searching
UniRef90 2021_04 (in the same archive) would close the member gap at several times the cost.

## 2026-11-05 (round 2) - Ruler v1.2 over the real 8M atlas and its three shuffled-weight controls

`real_units_20261105/` (`scripts/score_real_units.py` on an A10, us-east-1; calls per column `calls_*.json.gz`;
`summary.json`). These are positive-pole columns, 7,680 per atlas (6 layers x 1,280 units), K_r 256, rule v1.2
unchanged.

| atlas | context | structural | composition | none | insufficient |
|---|---|---|---|---|---|
| trained | 7,158 (93.2%) | **408** (5.3%) | 97 | 10 | 7 |
| shuffled 1 | 7,527 (98.0%) | 4 | 126 | 2 | 21 |
| shuffled 2 | 7,520 | 11 | 115 | 1 | 33 |
| shuffled 3 | 7,507 | 9 | 103 | 2 | 59 |

- *Context* (residue mode plus structural excess) is the default call for trained and untrained units alike. It
  does not separate them: the residue identity at the peak, plus the structure that residue brings beyond the
  composition-matched null, is enough to be called context. Every context call is "sequence-only not excluded",
  because the window baseline is not repaired (below).
- *Structural* calls (a structural descriptor with no residue mode) separate trained from untrained: 408 against
  4-11, about 50x. Their primary descriptors are state5 319, rama4 51, burial3 38. By layer: 16, 25, 76, 85, 130,
  76 (layers 0-5).
- *No association* is rare in both (10 against 1-2), so the procedure almost always finds something. A "none" call
  is a statement about this ruler's vocabulary, not about the unit (see the cases below: one "none" unit reads
  whole-protein composition; another drives two layer-2 units strongly, though what triggers it stays
  unidentified).

## 2026-11-05 (round 2) - Window baseline v1: the repair, against its pass criteria

`baseline_v1_20261105/` (`atlas/seqbaseline.py` v1; `scripts/ruler_challenges.py --baseline v1`; design, target,
support limits and pass criteria in `cases_round2_plan.md` section 2, with an amendment after the smoke recorded
there).

The two problems in v0, separated:
- **Expressiveness:** v0 multiplied independent per-site residue tables, so it could not represent residue
  conjunctions. v1 uses a small MLP on the one-hot +-5 window (11 x 21 -> 32 -> 1).
- **Level mismatch:** v0 fitted the window model on lower bins and tested it at the peak. v1 fits and tests at the
  same level: the density ratio between a column's peak tokens and background tokens.

**Target.** At the peak, the descriptor distribution that the +-5 window alone predicts, computed as background
tokens reweighted by the fitted peak/background density ratio of their windows. The test asks whether the
observed peak departs from it.

**Separation.** 5 folds by UniRef50 cluster: each fold's ratio is fitted on the other folds' clusters and applied
to its own.

**Support limits.**
- A call is "not estimable" below 40 peak tokens with a defined descriptor, or when a fold's effective sample size
  falls below 100.
- A fold trains on about 45 positives at K_r 64 and about 180 at K_r 256.
- The verdict is worded "unexplained by this fitted +-5 window model", never "structural".

| set | sequence-only types: "unexplained" / estimable calls | structural types: "unexplained" / estimable | not estimable |
|---|---|---|---|
| A, K_r 64 (old challenges) | G-x-G-x-x-G **0 / 5**; hydro_win **4 / 12** | 84 / 86 (helix 28/28, ordered 23/23, conj. exposed strand 12/12, buried 6/6, two_job 6/6, narrow 5/5, lys_helix 4/6) | 3 |
| E, K_r 256 (fresh challenges, seed 9) | hp_i34 0 / 10; kxxxe 1 / 6; pxxp 0 / 9; G-x-G-x-x-G 0 / 3; hydro_win **2 / 4** | strand_buried 12/12; helix_exposed 12/12; helix 4/4 | 0 |
| A, K_r 256 | pending | pending | pending |

**Against the criteria fixed beforehand** (sequence-only "unexplained" <= 1 per set; structural >= 80%):
- *Repaired:*
  - the motif failure (v0 called all 5 G-x-G-x-x-G calls "beyond the window"; v1 none);
  - the new sequence-only types (1 of 25);
  - power on structural columns (97-100%).
- *Not repaired:* hydro_win, the extreme of an additive +-3 hydropathy score, is "unexplained" in 4 of 12 and
  2 of 4. Its function is representable by the MLP, so this is an estimation failure at the extreme tail (ESS
  median 534-556), not an expressiveness one. Both sets therefore fail the <= 1 criterion.
- *Consequence for real units:*
  - "unexplained by this fitted window model" is evidence against a sequence-only account only for columns whose
    peak is not an extreme of an additive hydrophobicity-like window score;
  - for hydrophobic, TM-like or amphipathic columns, the verdict does not exclude sequence-only;
  - every structural or context call on a real unit below keeps **sequence-only not excluded**, and the cases test
    sequence against context directly, by intervention.

## 2026-11-05 (round 2) - The pair's balance: an unsuccessful explanation

`pair_balance_20261105/` (`scripts/pair_balance.py`; frozen test `pair_balance_frozen.md`, which records
everything inspected beforehand, the independent reviewer's alternative and an amendment after the smoke).

**Proposed computation (from discovery, 600 non-held-out proteins).**
- The pair detects "buried beta-strand position": its top tokens are 85% strand and 81% buried tercile, with V
  the commonest residue there.
- Their balance encodes the centre residue's identity and its N-terminal context: 86-heavy departures from
  the fitted relationship (h1200 = 1.11 + 0.66 h86, R^2 0.31) have centre D/S; 1200-heavy ones have K/G/S/Q
  at -1/-2.
- The opposed writes shift masked neighbours along a polar/hydrophobic axis, and this natural variation helps
  masked prediction.

**Reviewer's alternative.** The residual is a fixed linear direction applied to the layer-1 input, which is
dominated by the centre token's embedding. The D/S signature is then a property of the weights, not a computed
variable, and any "help" comes from neighbour composition.

**Fresh test** (600 held-out proteins with descriptors, excluding the first experiment's 400):

| observation | proposal predicts | alternative predicts | observed |
|---|---|---|---|
| 86-heavy departures with centre D/S | >= 3x base | also passes (fixed readout) | 23% vs 4% base (5.8x): passes, uninformative |
| gain from natural departures beyond injected ones, conditioning on neighbour polarity and centre D/S | > 0 | ~ 0 | **-0.00025 (95% CI -0.0019 to 0.0015)** |
| polarity slope (natural / injected) | - | equal | 0.0030 (CI -0.0011 to 0.0070) / -0.0017 (CI -0.0081 to 0.0048): both include 0 |
| median gain at masked neighbours | > 0 | ~ 0 | natural -0.000125, injected +0.000096 nats |
| \|change\| vs a same-norm change of another unit in the same masked forward | larger | similar | 0.0020 vs 0.0022 (natural); 0.0018 vs 0.0021 (injected) |
| +-3 window transplanted into other proteins: share above the unit's q99 (strand / helix / loop hosts) | high in strand | similar, and retained | 86: 0.20 / 0.11 / 0.13; 1200: 0.14 / 0.11 / 0.10 |

**Account.**
- *Supported:* the pair fires at buried strand positions on fresh proteins; departures from its fitted
  balance recur with centre D/S (5.8x); its single effects are opposed and add almost linearly (joint within
  25% of the vector sum, cosine -0.73 in discovery).
- *Contradicted:* "the natural variation in their balance helps prediction". Removing it, or injecting it,
  changes log p of the true neighbouring residues by ~1e-4 nats, no more than a same-norm change to an
  unrelated unit, and with no dependence on context or neighbour polarity that the intervals can see.
- *Neither explanation as stated:* the transplant shows that the units' top activations need more than the
  +-3 window (80-90% lost when it is moved), with a modest strand-host advantage. So "a fixed readout of
  local sequence" is too narrow, and "buried strand" is only weakly supported.
- *What is explained beyond naming what activates the units:* little. Their writes oppose and partly cancel by
  ordinary vector addition, and the balance tracks the centre residue. No downstream benefit of the balance is
  detectable at masked neighbours. The opposed output cosine remains a fact about the weights, not yet an
  explanation of anything the model does with it.
- *Not tested:* other readouts (the position itself when masked; later-layer readers; longer-range positions).
  The effects measured here are close to the noise floor of a single-unit intervention.

## 2026-11-05 (round 2) - A 650M "ubiquitin unit" that counts repeats

`ubq650_unit_20261105/` (`scripts/ubq650_probe.py`, `ubq650_repeat.py`, `ubq650_dose.py`, `ubq650_silence.py`;
predictions and their history in `ubq650_unit_predictions.md`). Chosen by the frozen rule: the 650M column with
the largest ubiquitin share of its top-64, **layer 15, unit 2087**, positive pole (share 1.0).

- **Discovery:** top-64 = His68 of ubiquitin copies in 8 polyubiquitin precursors (values 6.66-6.91). The peak
  reservoir is mostly not ubiquitin (H 31, P 21 of 64).
- **Pre-registered explanations:** family recognition, local His-window motif, structural patch. **All three
  failed their first probe:** in isolated human ubiquitin the unit reads -1.34 at H68 (between its q50 and q90),
  and likewise in NEDD8, mutants, shuffles and transplants. The highest value in isolated ubiquitin is 0.76.
- **Post-hoc hypothesis** (written after that failure, before testing): the response needs tandem copies.
  - Repeat dependence held: one copy fused to a ribosomal protein gives -1.4; UBB (3 copies) 3.3-3.8; UBC
    (9 copies) 5.7-6.7.
  - It also held beyond ubiquitin: SUMO1 x2 lifts the unit's maximum from 0.97 to 2.6; an unrelated lysozyme
    fragment x2 from 1.1 to 2.9.
  - Its direction ("already seen earlier") was wrong: copy 1 is as high as later copies.
- **Revised proposal and fresh test:** the unit's value grows with the number of identical copies of the local
  sequence in the protein. Ubiquitin x1-x9: -1.34, 2.14, 3.93, 5.02, 6.06, 6.68 (monotone, saturating). Five random
  held-out fragments x1 / x2 / x4: monotone in **5 / 5** (e.g. 0.39 / 3.09 / 4.54).
- **Downstream test (pre-registered):** silencing it at every position of Ub x4 changes log p(true) at masked
  copy-2 positions by -1e-5 nats, against -5e-5 for a write-matched control. The predicted drop was not seen. In a
  perfect repeat the masked residues are already near-certain, so the readout may sit at its ceiling; and one unit
  of 168,960 may be redundant.

**Account.**
- *Proposed computation:* repeat copy number of the local sequence within a protein, read in both directions.
- *What fresh inputs show:* copy number predicts the unit's value on held-out fragments it was never selected
  on (5 / 5 monotone).
- *What it explains beyond naming what activates it:* the ubiquitin enrichment is not family recognition. It is a
  repeat count, and polyubiquitin precursors are the most perfectly repeated sequences in Swiss-Prot. So the
  share of a unit's top-64 that is "ubiquitin" measures redundancy (as #7 suspected), and for this unit it says
  nothing about ubiquitin biology. A DMS comparison against this unit would be meaningless.
- *Not shown:* that the model uses the count. Silencing it moved nothing at the readout tried.
- *The earlier atlas-based inference* ("units whose top-64 are mostly ubiquitin copies") would have been read
  as ubiquitin-related without these probes.

## 2026-11-05 (round 2) - Six real 8M units: proposed computations tested on fresh proteins

`cases_round2_20261105/` (`scripts/case_units.py`, `case_inspect.py`, `case_tests.py`).
- **Predictions:** `cases_round2_predictions.md`, frozen per unit in commits dafee6e, 980b48e and b5c3283, each
  before that unit's fresh run. It records everything inspected beforehand, including the discovery-set runs of
  `case_tests.py`, which served as smokes.
- **Selection:** `real_units_20261105/selection.json`, frozen rule with seed 20261105 and fallbacks fixed before
  any example was seen.
  - 5 random picks within predefined ruler categories: structural x2, composition, context, none.
  - 1 deliberate pick: the "not detected" unit with the strongest next-layer reader (cosine 0.32). A deliberate
    pick exposes blind spots; it does not estimate how often arbitrary units are understandable.
- **Protein sets:** fresh = held-out with descriptors, excluding the first pair experiment's 400 (seed 1, 600). For
  the family unit, fresh = every Swiss-Prot sHSP never inspected for it, against all held-out non-members.
  Thresholds are each unit's discovery q99.9.
- **Standard readouts** (all six units):
  - the share of a predicted context at fresh top positions;
  - silencing at the masked top position: the signed change in log p of the predicted residues and of the true
    residue, against a unit matched on write magnitude in the same masked forward.

**Overview**
- First explanations failed on discovery material for 2 of 6 units: (5,1181) and (0,789).
- After revision, the frozen fresh predictions held:
  - in full for 4 units: (3,1238), (5,668), (0,767), (0,789);
  - in part for (5,1181);
  - not for (1,746), whose explanation failed on fresh proteins.

### (3,1238) random structural pick: recognition of the small-heat-shock-protein family

- **Proposed computation:** recognise the alpha-crystallin domain (sHSP family). The response needs the intact
  domain, more than a +-10 window, and is broadcast to most positions of the protein, including the N-terminal
  region outside the domain.

| fresh prediction | observed |
|---|---|
| family (146 never-inspected sHSPs) median fraction of positions above threshold >= 0.2; non-family (10,007 held-out) median 0, q95 <= 0.01; AUROC >= 0.9 | **0.54**; 0 and 0; **0.936** |
| members below 0.35 identity to every seed: median >= 0.05, below the >= 0.5-identity members | **0.36** (n 47) vs 0.63 (n 81); 0.54 at 0.35-0.5 (n 18) |
| CS-domain proteins (structural analogues of the domain, n 70): median <= 0.01 | **0** (max 0.003) |
| C-terminal 60% shuffled: N-terminal 40% response ratio <= 0.3; N-terminal 40% shuffled: C-terminal ratio >= 0.6 | **0.0**; **0.99** (30 members) |
| +-10 window around each member's maximum transplanted into 5 non-family hosts: share above threshold <= 0.1 | **0.007** (n 150) |
| composition shuffles: median 0, max <= 0.05 | 0, 0 (n 90) |
| standard: silencing lowers log p of P/T/S beyond the control; strand share at top above base | -0.065 (CI -0.113 to -0.010) vs control +0.090; strand 0.19 vs 0.14 (weak; 21 positions, few sHSPs among the 600) |

- **Account.**
  - *Supported:* a family-recognition computation that generalises beyond close relatives (identity < 0.35
    responds at about 60% of the level of close members). It does not respond to the domain's fold analogues.
  - *Beyond naming what activates the unit:*
    - the response is a property of the whole domain, not of any window inside it;
    - it is broadcast: positions outside the domain respond only while the domain is intact, so the unit's value
      at an N-terminal residue reports something about residues 60-150 positions away.
  - *Not shown:* any use of this signal. No masked-prediction test was frozen for this unit.
  - *Unexplained:* 20 held-out non-members respond at 10-26% of positions, mostly Arabidopsis proteins outside the
    seed-defined family. Whether they carry divergent alpha-crystallin domains was not checked.
  - *Ruler:* the structural call (state5, strand) was a by-product of the family's beta-sandwich. Sequence-only is
    not excluded by the baseline, but the transplants and shuffles show the unit needs the domain, not a window.

### (5,668) random "no association" pick: whole-protein composition

- **Proposed computation:** every position carries an estimate of the protein's overall amino-acid makeup along
  a GC-proteome axis: high G/A/R/P (and D/E), low F/Y/M/I/N/K. About half of it survives shuffling.

| fresh prediction | observed |
|---|---|
| Spearman(protein mean, GARP - FYMINK) >= 0.45 | **0.557** |
| eta2 protein >= 0.45; centre residue <= 0.05 | **0.573**; **0.016** |
| shuffle retention >= 0.35 (taxon-signature alternative <= 0.15) | **0.653** |
| chimera halves shift toward the partner by 0.25-0.75 of the gap (local alternative <= 0.1) | **0.502** (IQR 0.26-0.61, n 80) |
| standard: silencing lowers Q/H/E beyond the control | -0.127 (CI -0.149 to -0.107) vs -0.001 (CI -0.052 to 0.043) |

- **Account.**
  - *Supported:* a protein-level variable computed from the whole sequence and written at every position.
    Chimeras put each half about halfway toward its partner. It is mostly composition: 65% survives shuffling,
    and the rest depends on order in a way not identified.
  - *Beyond naming:* this is the ruler's blind spot made concrete. A per-position residue/structure test cannot
    see a variable that is constant along a protein. "No association" meant "nothing local".
  - *Not shown:* that the model uses it. Log p(true) at masked top positions: -0.016 (CI -0.054 to 0.017),
    control +0.007.

### (0,789) random context pick: a lone-cysteine readout (first explanation failed)

- *Unsuccessful explanation, on discovery proteins:* "C or S in an S/T-rich disordered context". Spearman with
  S/T count was 0.09, and replacing the +-3 neighbours with A kept 59% of top positions above threshold.
- **Proposed computation:** the centre C raises the unit; any other C within +-3 strongly lowers it (ridge
  weights -1.56 / -1.40 at -3 / +3). It separates isolated cysteines from CxxC / CC / CxC motifs.

| fresh prediction | observed |
|---|---|
| C has the highest rate | **2.55%** (next G 0.15%) |
| C ridge (fitted on discovery) R^2 >= 0.3 | **0.383** |
| lone minus clustered C median >= 0.8 | **1.37** (-0.65 vs -2.02; n 3,342 / 783) |
| neighbouring C -> A raises the value by >= 0.6 | **+0.72** (n 80) |
| S/T alternative: Spearman <= 0.2; +-3 -> A keeps >= 0.4 | 0.081; 0.50 |
| standard: silencing has no effect beyond the control | -0.0007 vs -0.0010: none |

- **Account.**
  - *Supported:* an approximately additive function of the input tokens at layer 0 (centre token plus neighbour
    terms), with one strong interaction: cysteines cancel each other.
  - *Beyond naming:* the "context" the ruler found (disorder, S/T) was a correlate. The unit's actual
    dependence is on nearby cysteines, which the ruler's vocabulary does not contain.
  - *Not shown:* any downstream use (silencing at the masked position does nothing beyond the control).

### (0,767) random composition pick: token identity plus an additive neighbour term

- **Proposed computation:** a fixed readout of the centre token (X > P > H, far above the rest) plus an
  approximately additive neighbour term through layer-0 attention: W and D raise it, K and R lower it.

| fresh prediction | observed |
|---|---|
| eta2 centre >= 0.4 | **0.515** |
| ridge R^2 (fitted on discovery) >= 0.25 for P and H | **0.365**, **0.369** |
| P and H the top two rates | P 2.4%, H 1.2%, all others 0.0% |
| X at rate >= 0.5 | untested: no X in the fresh proteins |
| standard: silencing lowers K/R/E beyond the control | -0.0041 (CI -0.0051 to -0.0032) vs 0.0000; tiny |

- **Account:** explained as a near-linear function of the input, nothing more. There is nothing beyond
  naming here, and that is the expected answer for a layer-0 unit.

### (5,1181) random structural pick: membrane-segment context (first explanation failed; revision partly supported)

- *Unsuccessful explanation, on discovery proteins:* "a hydrophobic segment read from sequence". Spearman with
  KD19 was 0.13; signal peptides never fired; values inside TM segments were flat across hydrophobicity bins.
- **Proposed computation:** raised over TM segments and their flanks, further raised within membrane proteins.
  It reads segment composition (order-insensitive) plus context from the rest of the protein.

| fresh prediction | observed |
|---|---|
| TM - no-TM protein >= 0.7; order TM > <= 10 > > 10 >= no-TM | **0.98**; -0.24 > -0.55 > -1.15 > -1.23 |
| buried helix outside TM: membrane minus soluble protein >= 0.4 | **0.25: fails** |
| AUROC protein mean (membrane vs not) >= 0.75 | **0.672: fails** |
| Spearman(h, KD19) <= 0.3 | 0.146 |
| TM >= buried helix (not TM) + 0.5 in every KD19 bin | 0.68, 0.59 hold; **0.37, 0.10 fail**. The top bin had already failed on discovery (0.01, n 110); I wrote the prediction inconsistently with my own discovery table. |
| TM segment inserted into soluble hosts: retention 0.15-0.65; shuffled in place >= 0.7 | **0.68: just outside**; 0.99 |
| standard: helix share at top above base; silencing lowers R/T/V beyond the control | 0.72 vs 0.37; -0.084 (CI -0.140 to -0.029) vs +0.013 |

- **Account.**
  - *Supported:* the unit is raised over annotated TM segments independently of local hydrophobicity, decays
    over about 10 residues away from them, and does not care about the order within a segment.
  - *Contradicted:* the protein-level part. On fresh proteins, membrane membership adds less (AUROC 0.67), and a
    TM segment keeps 68% of its value in a soluble host. The signal is more segment-local than I proposed.
  - *Unpredicted observation:* silencing it at fresh masked top positions lowers log p(true), -0.065 (CI -0.109 to
    -0.021), against -0.009 (CI includes 0) for the control. No prediction was frozen, so this is reported, not
    claimed.
  - *Beyond naming:*
    - a segment-level composition signal that is not hydrophobicity (Spearman 0.15) and is shuffle-invariant;
    - at the most hydrophobic windows, buried helices of soluble proteins reach TM levels, so whatever it
      reads overlaps with "very hydrophobic helix" there.
  - Sequence-only is not excluded: the baseline fails on exactly this kind of additive hydrophobic column.

### (1,746) deliberate pick ("not detected", strongest next-layer reader): the explanation failed on fresh proteins

- **Discovery evidence:**
  - signal peptides at 50x the base rate among top positions;
  - moving a signal peptide to mid-protein abolished it;
  - silencing changed layer-2 units 1006 / 455 by -0.43 / +0.36.
- **Proposed computation:** a signal-peptide detector with strong opposed outputs to two readers.

| fresh prediction | observed |
|---|---|
| SIGNAL rate >= 10x base and >= 3x TM rates | **0.04% vs 0.12% base: fails**; TM 0.65% / 0.57% |
| first-30 rate <= SIGNAL / 5 | 0.03% (moot) |
| moved signal peptides lose >= 80% | native share 0: uninformative |
| TM moved to the N-terminus stays <= 0.02 | 0.0 -> 0.007 (native 0: uninformative) |
| readers: 1006 down, 455 up, beyond the control, >= 5x | **-0.417** (CI -0.435 to -0.400) and **+0.333** (CI 0.319 to 0.347), against +0.005 and -0.012 |
| downstream over the signal peptide: no effect beyond the control (a predicted null) | +0.0002 (CI -0.0005 to 0.0008) vs -0.0005 |

- **Why the explanation failed:** on discovery proteins, the 46 top signal-peptide positions came from **2
  proteins** (Q91717 26, P02893 18; plus 2 in P0AD13) among 37 carrying a signal peptide. I counted tokens, not
  proteins. That lesson applies to every enrichment in this round, so the confirmatory tests here count proteins
  or use protein-level splits where it matters.
- **Account.**
  - *Supported:* the wiring. At the positions where the unit fires, it moves two specific layer-2 units in
    opposite directions, by 0.29 and 0.44 of their q50-q90 spans and 30-80x a matched write. Neither reader's own top
    activations involve signal peptides.
  - *Not supported:* any description of what it detects. On fresh proteins its top positions are mildly
    hydrophobic (61% I/V/L/F/A vs 33%) and TM-enriched (5x), and nothing else was found.
  - *Beyond naming:* nothing yet. The ruler's "none" stands, and so does mine.

### Across the six

- **Not detected by the ruler is not empty.** One "none" unit carries a protein-level composition variable; the
  ruler's per-position vocabulary cannot express it.
- **Ruler categories and causes disagree.** The family unit was called structural, the lone-cysteine unit
  context, the membrane unit structural. In each case the test that distinguished explanations was an
  intervention on the sequence (transplant, shuffle, insertion, neighbour mutation), not a descriptor.
- **"What have we explained beyond naming?"**
  - two non-local computations: family recognition broadcast across a protein, and a whole-protein composition
    estimate;
  - one interaction (cysteines cancel);
  - one segment-level signal;
  - two near-linear token readouts.
  - For none of the six did a frozen prediction establish that the model *uses* the unit for masked prediction.
    Silencing moves the predicted residues beyond controls for 4 units (moves predictions), and changes log
    p(true) beyond the control only once, unpredicted (5,1181).

## Spend

A10 compute 2026-11-03: ~53 billed minutes, ~$1.14 (marker test + four phase-1 launches, three of which
failed within 1-3 minutes on job bugs and were terminated by the driver); evening profiling job ~58 minutes, ~$1.25
(the controlling container restarted mid-run; supervision was re-attached and the instance terminated on completion).
2026-11-04: implementation A/B, ~56 billed minutes, ~$1.20 (us-west-1; new filesystem `esm-atlas-us-west-1`
holds the dataset copy, HF cache and this run's checkpoints and atlas, ~16 GB). Science pass on an A100 40 GB
(asia-south-1), ~102 billed minutes, ~$3.40 (new filesystem `esm-atlas-asia-south-1`: dataset, HF cache,
checkpoints, the 10.85 GB atlas). Follow-up on an A100 40 GB (us-east-1), 14:29-16:43 UTC, ~$4.4 (atlas copy #2 and the
held-out atlas on `esm-atlas-results`). Descriptors on an A10 (us-east-1), ~44 billed minutes, ~$0.95.
2026-11-05: workload comparison on an A100 40 GB (us-east-1), 01:44-02:45 UTC, ~$2.0; batch job on an A10 (us-east-1),
03:03-04:11 UTC incl. the rerun of two controls, ~$1.5; K sweep on an A100 (us-east-1), 04:00-04:23 UTC, ~$0.8.
Total ~$16.7.
2026-11-05 (later): strict held-out on an A10 (us-east-1), two attempts, 11:07-11:29 and 11:35-12:17 UTC, ~$1.4.
Total ~$18.1.
2026-11-05 (round 2): real-unit scoring on an A10 (us-east-1), ~17:04-17:46 UTC, ~$0.9. Total ~$19.0. No instance
is running.
