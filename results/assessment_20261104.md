# What the results establish, and what is still interpretation (2026-11-04)

Each claim is tagged **exact** (holds by construction for the activations the model computed, with
code and a test or run that checks it), **estimate** (a population quantity measured on a sample),
**model** (a prediction we compare against data), or **interpretation** (proposed, not established).
"For the activations the model computed" matters: bf16 forward, one GPU model and software stack. A
different GPU type can change activations in the last bits; the atlas is exact for what was computed.

## 1. Collector vs ruler

The collector preserves specified summaries; the ruler (not run yet) asks whether they explain the model.

| quantity | status | definition and code | evidence |
|---|---|---|---|
| top-K per (layer, pole, unit) | exact | K largest s = +h / -h; ties to the smaller token id (`atlas/core.py:66` top_key; admission s >= tau at `:254`; key test `:501`; dense path `:407` takes every row tied at the K-th value) | `tests/test_exactness.py` incl. tie-heavy data in shuffled order; A100 gate: reference vs new bitwise (`results/science_650m_20261104/gate_compare.json`) |
| reservoir per (layer, pole, unit, bin) | exact | K_r smallest keys of a bijective 32-bit hash among tokens in the bin (`:46` token_keys; `:256` admission; `:547` bulk rows) | same tests; seed verification 0 of 1,666,065 short (`followup .../full/run.json`) |
| bin edges | estimate (as quantiles) | exact quantiles of a 262,144-token pilot (`atlas/reference.py:55`); the q99.999 edge is about the 3rd-largest pilot value, so its population quantile is noisy | definition only; bin masses below measure it |
| bin masses, sums, sums of squares, enrichment counts, Gram | estimate | exact counts and sums on the hashed 5% subsample (`atlas/stats_triton.py:26`, `:121`) | kernel tests vs torch (`tests/test_stats_triton.py`); gate: v1 vs v3 sums agree to 1e-8 |
| K/(n+1) candidate rate | model | expected top-K admissions per element under exchangeable order | section 2 |
| "job", "coherence", families, held-out generalisation | interpretation | not computed | none yet |

## 2. Filtering

Exactness does not use random order: the running K-th key only rises, so `s >= tau` (and `s > seed`)
admits every token that can still enter, whatever the order (`:254`, `:501`). Seeds are exact
because the top-K seed is just below the 2K-th pilot value and the pilot tokens are streamed. K/(n+1)
is the expected admission rate under exchangeable order; our compaction threshold is
min(top-K tau, reservoir floor), so compacted elements include reservoir candidates, and the logged
top-K candidate fraction counts candidates against the batch-start K-th key (entries are fewer).

Observed / predicted (`topk_cand_frac` / `pred_topk_frac_iid` per batch, filtered batches):

| run | batches | median [p10, p90] |
|---|---|---|
| A10, unseeded, admit s > tau, K_r 16 | 0-50 / 50-200 / 200-620 | 1.02 / 0.96 / 0.90 [0.78, 1.16] |
| A100, seeded, admit s >= tau, K_r 64 | 0-50 / 200-1000 / 3000-6143 | 1.05 / 0.95 / 0.95 [0.77, 1.21] |

Where it fails: short-protein batches (Lmax < 150) 1.38-1.46x; long batches 0.84-0.89x (tail mass
concentrates in a minority of batches); single batches with a compositionally extreme protein 11-59x
(A10 batch 111, the poly-Ser Dictyostelium protein; A100 batches 4095, 4719, 4432). The tie change
(s > tau to s >= tau) cannot be read from these runs: the batch orders differ.

## 3. Costs

Steady state, A100, device ms per batch at batch 3000 (`results/followup_650m_20261104/full/profile_batch_3000_ranges.txt`):

| part | ms |
|---|---|
| compaction (one read of H) | 18.4 |
| subsample statistics (stats kernel / Gram / int8 enrichment) | 58.1 (18.9 / 24.4 / 10.6) |
| reservoir bulk rows | 10.5 |
| candidate classification + top-K merge + reservoir merge | 4.5 (0.8 + 1.1 + 2.6) |
| idle gaps in the merge (syncs, launches) | ~7.9 |

Per-batch regression on 244 clean paired batches: excess = 24.5 ms + 21.0 ms per 10k tokens + 22.2 ms
per 1M compacted elements (R^2 0.71); the median batch compacts 0.29M elements, about 6 ms of a 101.5
ms median excess. Dense recovery: 1.0% of the A100 pass. At steady state the remaining cost is fixed
per-token work, not candidates.

Where the gains came from, by measurement:

| change | gain | mechanism |
|---|---|---|
| threshold filter vs per-batch topk | A10 phase 1, top-K only: naive 85% vs torch filter 38% | fewer candidates |
| Triton compaction vs torch filter | 38% vs 13% | fusion |
| stacked merge | -32 ms per batch; idle 8.5 -> 0.9 ms | fewer launches and syncs |
| stats v3 | 333 -> 58 ms | removing materialisation (fusion) |
| prefetch | 8.4 min -> 0 | host overlap |

The current traces do not answer "candidates vs fusion" for the complete workload: no run has the
naive method doing the whole collection (top-K, reservoirs, statistics), and candidate counts were
never varied with everything else fixed.

## 4. Related work (inspected)

| reference | closest operation | same as ours | different |
|---|---|---|---|
| baukit `runningstats.py` | `TopK`: per batch, a full per-column topk into a 5K buffer, recompressed when full; `Quantile` (KLL), `Covariance`, `CrossIoU` | exact streaming top-K values; online moments; thresholded co-occurrence | no threshold filter (every batch fully selected: our "naive" path); torch tie order; `Reservoir` samples whole rows by fresh randn scores (not per unit or bin, not reproducible); quantiles approximate |
| SAE-Vis `data_fetching_fns.py` | top-k over a fixed, fully materialised activation tensor; random samples from equal-width value intervals [0, max] | top examples plus samples per activation range | not streaming; value-range intervals, not quantile bins; random, not keyed; selected features only |
| Dr. Top-k (arXiv 2109.08219) | delegate-based filtering: subrange maxima set a threshold, exact top-k on survivors | threshold filtering to cut selection work | one array, one query; threshold from the same data, not from stream state |
| NVIDIA GVR V2 | threshold admission, bounded candidate capacity, exact recovery, candidates = K + boundary ties + shell | our filtered path, overflow, dense redo; V1's temporal hint is our streaming tau / pilot seed | one row per step; ties "exact value multiset, positions may differ" (our old policy); fp32 scores |

A fair comparison needs: the same shapes (32k tokens x 5,120 units x 33 layers x 2 poles), bf16
inputs, the same output definitions (exact values and ids under one tie policy; bottom-k per bin with
one hash), streaming state, the complete collection workload, the same GPU, and overhead measured
against the same forward. The comparison that would most change our understanding: baukit's TopK +
Covariance plus a naive per-bin reservoir, doing the complete workload on the same stream. If it lands
near our overhead, the filter is not what makes the job cheap. The kernel-level one (GVR V2's operator vs
our compaction on one batch with a prior threshold) is secondary.

## 5. The unit itself (8M, 300 random proteins, CPU)

| | pre-GELU h | GELU(h) | written norm vs residual | silencing effect |
|---|---|---|---|---|
| layer 2 unit 826, positive extreme (Asp) | 18.0 | 18.0 | 87.6 vs 82.4 | log p(Asp) -2.80; Leu/Phe/Ile up |
| same unit, negative extreme | -20.5 | -0.0000 | 0 | 0 |
| same unit, h nearest -0.75 | -0.78 | -0.170 | 0.83 vs 14.7 | 0.0055 |

Across units, GELU at the q0.0001 (most negative) level: median |GELU| 0.003-0.008 (layers 0-2),
0.0000 (layer 5). GELU is bounded below by -0.17, so at any model size the negative pole's extremes are
where the unit writes nothing: a suppressed response. The weak opposing write lives near the median.

## 6. Hook points

InterPLM's SAEs and its neuron baseline both use "hidden representations ... after transformer block
layers 1 through 6" (block outputs, 320 coordinates for 8M); neurons were scaled to [0, 1] and passed
through an identity "SAE". We collect FFN intermediate units (1,280 per layer, pre-GELU). Not like for
like; its "up to 46 neurons per layer" is about a non-privileged basis.

## 7. Independence

- Ubiquitin: 50 Swiss-Prot entries contain human ubiquitin exactly (155 copies), 148 carry a near
  copy (~23k near-identical residues in the stream), 5 of them held out; ubiquitin is in ESM-2's
  training data. Unit samples that touch ubiquitin are mostly copies.
- The eLife DMS (Mavor et al. 2016) uses yeast ubiquitin, which differs from human at 3 positions
  (19 P->S, 24 E->D, 28 A->S, from our sequences), under several chemical growth conditions.
- Date holdout: at least 25.0% of the 10,010 held-out entries share a current UniRef50 cluster with an
  older Swiss-Prot entry (a lower bound: TrEMBL relatives not counted). It establishes later entry,
  not novelty; 2,802 of the 10,010 have no AlphaFold model.
