# Measurement #4: four relations between units, and where they disagree (written 2026-11-05, before the 8M atlases exist)

`scripts/families.py` on the 8M atlases of job `batch_20261105` (pretrained; three shuffled-weight controls),
positive pole, all pairs of units within a layer (1280 units, 818,560 pairs per layer, 6 layers):

| relation | definition | what it measures |
|---|---|---|
| co-activation | Pearson r of pre-GELU h over the hashed 5% subsample (atlas Gram and sums) | the bulk of the distribution, mostly near the median |
| input weights | cos(W_in[i], W_in[j]) | what the units read |
| output weights | cos(W_out[:, i], W_out[:, j]) | what the units write |
| tail overlap | Jaccard of the token sets above q99.9, from the bottom-256 reservoir sketches (one hash over all units, so the bottom-k estimator applies) | which tokens drive both to their extremes |

Earlier caution (assessment): the Gram is roughly input-weight similarity seen through the input covariance,
so co-activation and input weights should agree with each other more than either agrees with tail overlap.

| quantity, per layer (pretrained) | predicted |
|---|---|
| Spearman(co-activation, input cos) | 0.3-0.7 |
| Spearman(co-activation, tail Jaccard) | 0.05-0.3 |
| Spearman(input cos, output cos) | -0.1 to 0.2 |
| Spearman(co-activation, output cos) | -0.1 to 0.2 |
| top-5 neighbour agreement, median over units: co-activation vs input | 10-40% |
| ... co-activation vs tail | 5-25% |
| ... input vs output, tail vs output | < 5% each |
| pairs with tail Jaccard >= 0.5 | 0.01-0.5% in layers 0-1, fewer later |

Shuffled controls: co-activation vs input cos stays in 0.3-0.7 (two linear read-outs of a shared input
correlate whether or not they were trained); input vs output ~0; tail overlap concentrated on residue
identity at every layer (shuffled embeddings still separate the 20 residues), so near-duplicate tails
(Jaccard >= 0.5) at least as common as in the trained model's layer 0.

Disagreements to write up (chosen after looking, three concrete pairs, with their top-K residues and the
values of all four relations): (i) same tail, different writes; (ii) high co-activation, disjoint tails;
(iii) near-identical input weights, different tails. A family defined by one relation is not a family by
another unless these cases are rare; the numbers above say how rare.
