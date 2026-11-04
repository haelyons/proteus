# followup_650m: predictions (written 2026-11-04, before the run)

Job `jobs/followup_650m` on one A100 40 GB (us-west-2, same GPU model as the science pass), same frozen
config as science_650m except two changes: (a) batches and their host-side contexts built on a prefetch
thread, (b) stats-kernel row split (R automatic: ~4 programs per SM, R = 6 at D = 5120 on 108 SMs).

## The two changes

| quantity | science pass (observed) | predicted after |
|---|---|---|
| stats kernel, ms per batch (33 layers) | 33-36 | 7-11 |
| gap between batches (host batch building), mean | 0.119 s | 0.005-0.03 s |
| host batch building over the pass | 8.4 min | 0.3-2 min |
| make_ctx inside the atlas pass | 11 ms | 1-3 ms (device copy only) |
| overhead, last 1,600 batches, median / mean | 0.271 / 0.273 | 0.21-0.23 / within 0.01 |
| overhead, whole run, median / mean | 0.286 / 0.296 | 0.23-0.25 / within 0.015 |
| stream wall (first batch start to last batch end) | 77.0 min | 64-68 min |
| end to end (pilot + stream + save) | 82 min | 69-73 min |

Stream budget after: forward 48.0 + atlas 11.5-12.5 + host gaps 0.3-2 + checkpoints 3.8 + paired timing
forwards 1.9. This does not reach 60 min. The remaining scheduled costs are configuration, not code:
paired timing forwards every 25 batches (1.9 min) and checkpoints every 1,500 batches (3.8 min); without
both, 59-62 min. Risk: the prefetch thread holds the GIL while building a batch (~0.1 s per 0.6 s
batch); if kernel launches starve, the atlas pass lengthens (0-10 ms per batch expected).

## Checks

| check | prediction |
|---|---|
| GPU tests incl. row split at the 650M shape | pass |
| gate, 60 batches, v1 + loop vs v3 (row split) + stacked | bitwise equal except sums (< 1e-7 relative) |
| atlas #2 exact fields vs atlas #1 (science pass) | identical (same GPU model, deterministic forward and ties; the row split changes only float sums) |
| redo batches | 40-55, by batch ~2,000, layers 0-1 |
| nvidia-smi logger | the process is gone within a few minutes; smi.err says why |

## Data for the ruler (no coherence computed)

| quantity | prediction |
|---|---|
| held-out atlas | 10,010 proteins, 4.08M tokens, ~130 batches, 2-4 min |
| top-64 lists, pole 0: median distinct UniRef50 clusters | 10-25; lists with < 10 clusters 25-50% |
| q99.999 bins, pole 0: median clusters; bins with < 10 | 25-45; 3-15% |
| q99.99 bins: median clusters; bins with < 10 | 45-60; 1-5% |
| q99.9 and lower bins: bins with < 10 clusters | < 1% |
| AFDB descriptors, whole build (575,748 accessions) | ok 95-97%; no model 3-4% (mostly viral); sequence mismatch 0.1-0.3% |
| residues with pLDDT < 70 / < 50 | 15-18% / 9-12% (2,000-protein sample: 16.3% / 10.2%) |

Cost: ~3-3.5 h on gpu_1x_a100_sxm4 at $1.99/h, ~$6-7 (cap 5 h: at most $10).
