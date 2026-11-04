# science_650m: predictions (written 2026-11-04, before the run)

Job: `jobs/science_650m`, ESM-2 650M, bf16, one GPU, first with capacity of A100 40 GB (SXM4, PCIe),
H100 80 GB (SXM5, PCIe). Swiss-Prot 2026_03 minus the 10,010 entries created after UniProt 2021_04
(17 Nov 2021; ESM-2 trained on UniRef50 2021_04). Frozen config: K=64, Kr=64, six bin edges at pilot
quantiles 0.5 / 0.9 / 0.99 / 0.999 / 0.9999 / 0.99999 (pilot 262,144 tokens), seeded thresholds,
rho_sub 0.05, rho_res 0.01, top-K ties broken by the smaller token id, stats v3 (Triton), merge stacked.

## Tests and equivalence gate

| check | prediction |
|---|---|
| GPU tests: compaction kernel, CUDA exactness (v3 now default), stats/Gram kernels at n=1600, D=5120 | pass |
| stats kernel time per layer (A100 / H100) | 0.10-0.25 ms / 0.07-0.15 ms |
| Gram kernel, upper triangle, per layer (A100 / H100) | 0.20-0.35 ms / 0.10-0.20 ms |
| gate, 60 batches: A (v1 + loop) vs B (v3 + stacked) checkpoints | bitwise equal on every exact field, top-K ids included; sums rel diff < 1e-6 |

## Full pass

| quantity | A100 40 GB | H100 SXM5 |
|---|---|---|
| tokens | 194.2M (198.26M - 4.08M held out) | same |
| batches | ~6,090 | same |
| hook-free forward per 32k-token batch | 0.40-0.55 s | 0.20-0.30 s |
| overhead (paired batches, excl. redo / profiled / flush / checkpoint), median | 0.10-0.18 | 0.12-0.22 |
| mean minus median | < 0.02 | < 0.02 |
| stream wall (run_atlas, excl. load, pilot, final save) | 55-75 min | 30-45 min |
| pilot (262k tokens, edges, seeds) | 2-4 min | 1-3 min |
| checkpoint (~4.3 GB) / final atlas.pt (~11 GB) | ~55 s / 2-4 min | same (network filesystem bound) |
| redo batches | < 1% of batches, mostly in the first ~120 and on compositionally extreme proteins | same |
| seed verification | 0 seeded bins short; 0 top-K groups short | same |
| q99.999 bin mass (subsample estimate), median over units | 0.3e-5 to 3e-5 (edge = 3rd largest of 262k pilot values) | same |

Why the overhead falls from 0.288 (A10, 2026-11-04): subsample statistics 333 ms per batch on the A10
become one pass (~10 ms), one int8 GEMM (~13 ms) and a half-size bf16 Gram accumulated in place (~20 ms).
Kr=64 keeps the q99.99 bin above the bulk-row cutoff, so compaction admits ~2e-4 of elements (vs ~3e-5
with Kr=16): more survivors, fewer early exits in the compaction kernel, +10-20 ms per batch on the A10.
Top-K ties at tau now enter as candidates (+~10% top-K candidates).

Billed estimate: A100 ~95 min, ~$3.2 (cap 4 h: at most $8); H100 SXM5 ~60 min, ~$4.3 (at most $17).
