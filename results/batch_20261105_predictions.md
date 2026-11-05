# Job `batch_20261105`: predictions (written 2026-11-05, before it runs)

`jobs/batch_20261105/run.sh`, one A100 40 GB (us-east-1, where atlas #2 is). Cap 2 h, expected ~60 min, <= $4.

## A. 650M tie audit (filtered top-K vs baukit's algorithm, first 60 batches)

Measurement #1 found equal values and ids differing at 56% of 21.6M slots (predicted: hundreds to a few
thousand). The 8M bf16 check put every difference in the tie group at the K-th value (0 above it) and
showed 71% of slots inside exact ties; fp32 had 0 differences. bf16 has 8 mantissa bits, so at 650M:

| quantity | predicted |
|---|---|
| values equal | yes |
| ids differing above the K-th value, after canonical sort | **0** (anything else means one side is not an exact top-K) |
| slots inside exact-value tie groups | 50-80% |
| ids differing at the K-th value (canonical) | 1-5% of slots |
| columns whose K-th value is tied with a token outside the list | 50-80% of 337,920 |

## B. #8 K sweep at 650M (complete workload, first 200 batches, full-stream seeds)

Reference: the follow-up's own first 200 batches (K = 64): overhead median 0.26 on batches 100-199 (4 paired),
20 redo batches (layer 0); compacted top-K fraction 1.39e-5, reservoir 2.16e-5. The cost regression
(22.2 ms per 1M compacted elements) puts top-K candidates at ~3.4 ms per batch at K = 64 there, ~0.9 at
K = 16 and ~14 at K = 256, plus a top-K merge that grows with K (every column is touched every batch early).

| K | overhead, batches 100-199, median | top-K candidates per batch (100-199) | redo batches |
|---|---|---|---|
| 16 | 0.24-0.27 | ~40k (0.25x K=64) | 15-30 |
| 64 | 0.25-0.28 | ~150k | 15-30 |
| 256 | 0.28-0.40 | ~600k (4x) | 15-30 |

Marginal cost: 15-60 ms per 1M top-K candidates, merge included. If K = 256 costs more than 0.40, the merge
(sort width K) and not the candidate count is the cost, and the "fixed per-token work" account of the
steady state does not carry over to large K.

## C. 8M atlases (K 64, K_r 256, bf16, Gram on)

| run | predicted |
|---|---|
| pretrained, full stream (194M residues, ~6,100 batches) | stream 8-15 min (host batch building bound, ~80 ms/batch); seeds 0 short; overhead median 1-4 (the 8M forward is tiny, the per-token statistics are not) |
| held-out, same edges | 4.08M residues as at 650M; < 1 min |
| three shuffled controls, concurrent | 10-20 min together; each streams the same tokens as pretrained (n_seen equal) |
| atlas.pt / slim.pt | 0.61 GB / ~0.21 GB each |

## D. #7 ubiquitin share of unit samples

Copies of human ubiquitin (>= 70/76 identical, ungapped) cover 29,944 residues in 146 proteins (155 exact
copies in 50 proteins; 5 proteins held out): ~0.015% of the stream. Under no association a unit's top-64
would hold a copy residue with probability ~1%. Copies at the same position in identical context give
near-identical activations, so a unit that reaches into ubiquitin at all should fill its list with copies.

| quantity | 650M (337,920 columns) | 8M pretrained (15,360) | 8M shuffled (each) |
|---|---|---|---|
| columns with any copy residue in top-64 | 2-6% | 2-6% | 2-8% |
| columns with >= half of top-64 in copies | 0.1-1% | 0.1-1% | similar to pretrained or higher |
| peak reservoir bins with >= half in copies | 0-50 columns | 0-10 | 0-10 |

The shuffled column is the point: if shuffled weights also put ubiquitin copies at the top, top-K
concentration on ubiquitin reflects redundancy in the stream, not something the trained model computes.

## Addendum before launch (02:57 UTC)

No A100 or H100 capacity in any region at 02:55; an A10 (24 GB) is free in us-east-1, where atlas #2 is. This
launch runs parts A, C and D (`parts.txt`); B (the K sweep) waits for an A100 so its timings stay comparable
with the A100 references above. A's outputs and D's do not depend on the GPU. C's times were predicted for an
A100: on the A10 the 8M forward and statistics are ~2-3x slower, so pretrained 10-20 min and the three
controls 12-25 min together, unless the host side still dominates, in which case the A100 numbers stand.
