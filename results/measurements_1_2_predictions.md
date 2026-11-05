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

## #2b Ruler v1.1 on a fresh noise draw (written 2026-11-05 after the v1 gate, before this draw exists)

v1 failed its gate on trp (9/12), lys_helix (9/12), two_job (9/12) and ordered (5/12); observed values, the
token-id defect found first (proteins > 1,024 residues) and the gate-script bucket fix are in results/README.md.
The 162 columns that showed this are now the design set; v1.1 (`atlas/ruler.py`, `--rule v1.1`) changes three
things, each traced to a failure there, and is scored on a fresh draw: same 60,000 proteins, `--seed 1
--noise-seed 1` (new noise per residue, new pilot, new reservoir hash), gate unchanged (`scripts/eval_synthetic.py`).

1. Primary descriptor p x 3 (it is the smallest of three p-values; both v1 false positives were structural).
2. Composition: residues tested against P(a | state) on the primary descriptor, not the background; a residue mode
   is p < 0.01 and log2 >= 1. (v1's effective number <= 4 missed W at 33-47% of the peak, 25-36x background.)
3. Several modes: the second state is tested among samples outside the top mode (conditional), not on the
   marginal (a two-mode sample splits between its modes; secondary log2 0.13-0.44 in v1's misses); an ordinal
   descriptor's adjacent level does not count.

| column (alpha >= 1, 12 each) | predicted pass, v1.1 |
|---|---|
| helix | 12/12 (one mode, to q90) |
| buried | 10-12/12 (risk: residue excess within the buried tercile at its extreme gives "context") |
| trp | 12/12 composition |
| lys_helix | 12/12 context |
| two_job / ordered | 11-12/12 each, several modes; still indistinguishable from each other |
| narrow | associated 12/12; "peak only" 7-10/12 |
| hydro_win | 12/12 associated; mostly context now (hydrophobics beyond state), primary state5 or rama4 |
| noise and every alpha = 0 (42) | >= 41/42 |

alpha = 0.5 (descriptive, 3 each): trp and lys_helix now detected in >= 2/3 (W ~10% of the peak against 1.3%);
narrow 0-1/3; helix, buried, ordered, two_job 3/3; hydro_win 1-2/3.
v1.1 is also scored on the design set (in-sample, reported as such). Neither result is a licence to score real
columns: that waits for sign-off on the rule, as v1 did.
