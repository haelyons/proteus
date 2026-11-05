# Measurements #1 and #2: predictions (written 2026-11-05, before either runs)

## #1 Complete workload: filtered vs naive vs baukit's top-K algorithm (`jobs/workload_ab`, ESM-2 650M, A100)

60 batches each; overhead = atlas pass / hook-free forward - 1, every batch paired. Prefix = first 60
batches, fresh state; steady = batches 4500-4559 (filtered resumed from the follow-up's checkpoint; naive and
baukit stateless, fresh). baukit = its TopK algorithm re-implemented (installing the package was refused
here); top-K only (it has no per-bin reservoirs).

| run | prediction |
|---|---|
| complete, filtered vs naive, outputs | bitwise equal on every field (same statistics kernels on the same rows), top-K ids included |
| top-K only, filtered vs naive | equal |
| baukit vs filtered, top-K values | equal; ids differ only at ties (hundreds to a few thousand of 21.6M slots) |
| overhead, complete filtered: prefix / steady | 0.28-0.38 / 0.20-0.22 |
| overhead, complete naive (prefix and steady alike) | 8-20 |
| overhead, top-K only: filtered (prefix) / naive / baukit | 0.05-0.12 / 0.3-0.6 / 0.3-0.5 |

Reading: complete naive / complete filtered atlas time = the gain from not selecting every element
(predicted 40-80x); naive top-K vs baukit = two non-filtering top-K implementations (predicted within 2x of
each other); filtered top-K vs either = the filter on the selection alone.

## #2 Synthetic columns through the atlas and ruler v1 (`scripts/synthetic_columns.py`, `atlas/ruler.py`)

~60,000 random non-held-out proteins with AlphaFold models (~21M residues), 9 column types x alpha in
{0, 0.5, 1, 2, 3, 4} x 3 noise replicates = 162 columns, K = K_r = 64, the 650M build's edge quantiles.
The ruler reads only the atlas; calls are joined to column types afterwards (`scripts/eval_synthetic.py`).
Gate counted at alpha >= 1 (12 columns per type); alpha = 0 columns count as noise.

| column | expected call | predicted pass (alpha >= 1) |
|---|---|---|
| helix | structural, state5, one mode (helix) | 12/12; persistence to q90 at alpha >= 1 |
| buried | structural, burial3, one mode (buried) | 9-12/12 (risk: state5 picked as primary, both p at the Monte Carlo floor) |
| trp | composition | 12/12 |
| lys_helix | context (Lys, helix) | 12/12 |
| two_job | several modes (helix and strand) | 10-12/12 |
| ordered (one meaning, two contexts) | several modes, indistinguishable from two_job | 10-12/12 |
| narrow (N-cap S/T/D/N at the peak, any S/T weaker) | associated at the peak; structural mode peak only | associated 12/12; "peak only" 6-9/12 (at alpha 3-4 the q99 bin may still hold N-caps: "to q99") |
| hydro_win (sequence-only) | associated, burial or state | 12/12 (the instructive case: a sequence-only column looks structural) |
| noise and every alpha = 0 column | no association found | >= 69/72 |

At alpha = 0.5, weak residue jobs will be missed by design: "composition" requires the peak's residues to
concentrate on <= 4 types (effective number), and a Trp column at alpha = 0.5 puts only ~10% Trp in the peak.
Predicted: trp alpha = 0.5 called "no association" in >= 2 of 3; common indicators (helix, ordered,
two_job, buried, hydro_win) detected at alpha = 0.5 in >= 2 of 3.
