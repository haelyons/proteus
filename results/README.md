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

## Spend

A10 compute 2026-11-03: ~53 billed minutes, ~$1.14 (marker test + four phase-1 launches, three of which
failed within 1-3 minutes on job bugs and were terminated by the driver).
