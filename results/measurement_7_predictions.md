# Measurement #7: ubiquitin audit and DMS alignment (written 2026-11-05, before either is computed)

Part 1 (share of each unit's samples that are ubiquitin copies): predictions in `batch_20261105_predictions.md` D.

Part 2, alignment of the deep mutational scan (Mavor et al. 2016, eLife 15802; per-allele fitness CSVs from
github.com/fraser-lab/pubs2014, `scripts/dms_align.py`):

| quantity | predicted |
|---|---|
| wild type of the scan | yeast ubiquitin; differs from human at exactly 19 (P->S), 24 (E->D), 28 (A->S) |
| alleles | 76 positions x (20 amino acids + stop) = 1,596 rows; >= 90% of the 1,444 missense alleles scored under DMSO |
| conditions | DMSO (reference), caffeine, DTT, hydroxyurea, MG132 |
| fitness of the humanising substitutions S19P, D24E, S28A (DMSO) | near wild type (> -0.1 on the scan's scale, where stop ~ -0.5) |
| allele fitness, Pearson between DMSO and each condition | 0.6-0.9; the lowest is the condition the paper calls most perturbing |
| positions whose mean fitness moves by > 0.1 between conditions | 5-20 of 76 |

What this decides: a unit-level check against this scan compares human-sequence activations with yeast
fitness, under five conditions that disagree at some positions; the scan is one organism's tolerance, not a
label of what a position "does". The audit says how many positions that leaves usable as condition-stable.
