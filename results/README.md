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
Bonferroni p < 0.01, and 7 are stable at p > 0.1 (8, 13, 33, 42, 44, 54, 67). That rests on three-replicate SDs
being honest errors; if they understate the noise by 2x, Q is 4x too large. Either way the prediction (5-20)
was too low.

What it decides: a unit-level check against this scan compares human-sequence activations with yeast
fitness (the three differing positions are themselves near-neutral in yeast), and most positions' tolerance
moves with the condition. A comparison should use positions stable across conditions, or name the
condition; "the DMS says this position matters" is not one statement here.

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
Total ~$16.7. No instance is running.
