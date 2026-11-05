# Mechanistic round 1: valine pair, travel, ubiquitin (written 2026-11-05, before any of it runs)

ESM-2 8M, fp32, CPU (`scripts/pair_silencing.py`, `scripts/travel_v2.py`, `scripts/ubq_dms_scores.py`).
"Fresh proteins" = the date-held-out entries (never in the stream the pair was selected from).

## 1. Layer 1, units 86 and 1200

Facts from the weights (design inputs, not outcomes): input cosine 0.81, biases -0.036 / -0.090, |w_in| 1.68 /
1.97; output cosine -0.661, the most opposed write to unit 86's among all 1,280 units of the layer; |w_out|
1.68 / 1.12. Projected on 86's write, 1200's write is -0.74 per unit of activation, and it has a component of
~0.84 orthogonal to it. So when both fire, about half of 86's write along its own direction is cancelled and
an orthogonal part is added; when their activations differ, the pair writes something neither does alone.

Design: 400 random held-out proteins. Positions: "both" (both units above their q99 on these proteins), and,
where they exist, "86 only" / "1200 only" (one above q99, the other below its q90). At each position, one at a
time: silence 86; silence 1200; silence both; reduce both to half (post-GELU). Controls matched per position:
for each of 86 and 1200, the layer-1 unit (not 86 or 1200) whose write magnitude |GELU(h)| x |w_out| at that
position is closest (ties broken by activation); the two matched units form the control pair. Readouts: logits
at the position (unmasked); masked-residue predictions at a random 15% of the other positions; layer 2-5
pre-GELU activations at the position.

| quantity (median over "both" positions unless stated) | predicted |
|---|---|
| Pearson r of h86, h1200 over all tokens of the fresh proteins | 0.80-0.95 (0.91 on the training stream) |
| top-64 of each on the fresh proteins that are V | >= 60% |
| cosine between the logit changes of silencing 86 and of silencing 1200 | -0.3 to -0.8 |
| sub-additivity: \|D(both)\| / (\|D(86)\| + \|D(1200)\|), logits | 0.3-0.7 for the pair; 0.8-1.0 for matched control pairs |
| KL of silencing both < KL of silencing 86 alone | in >= 50% of positions |
| Spearman of layer-2 activation changes, silence 86 vs silence 1200 | -0.3 to -0.9; matched controls -0.2 to 0.2 |
| halving both / silencing both, logit change norm | 0.4-0.6 (near-linear) |
| which residue predictions move when 86 is silenced | V down, I / L / A up at the position (the write is valine-related); sign reversed for 1200 |

If the pair's joint silencing is about additive, or its effects are no more opposed than the matched
controls', the opposed write cosine does not translate into opposed effects and "a cancelling pair" is not
the explanation. That outcome is reported as such.

## 2. Travel, re-measured

Same 300 units and positive-extreme tokens as measurement #3 (same seed), plus per case: the KL at every
other position (not only the maximum), by distance from the silenced position; a control unit at the same
position matched on write magnitude (closest |GELU(h)| x |w_out| among the other units of the layer); and a
masked task (a random 15% of the other positions masked; change in log p of the true residue there).

| quantity | predicted |
|---|---|
| median over positions j != i of KL_j, relative to KL at i | 1/1000 to 1/100 (the earlier "maximum" was a tail) |
| sum over j != i of KL_j, relative to KL at i | 1-10x in layers 0-3; < 1x in layer 4; exactly 0 in layer 5 |
| median KL at \|j - i\| <= 3 / median KL at \|j - i\| > 16 | >= 10 |
| target / matched-control ratio of the summed travel | 1-3 (travel follows write magnitude, not unit identity) |
| masked task: median \|D log p(true)\| at masked positions, layers 0-4 | 1e-4 to 1e-2; 3-30x the unmasked change at the same positions |
| layer 5, masked task | exactly 0 at every other position |

## 3. Ubiquitin

- DMSO is the primary condition; the other four are reported separately, never pooled.
- The seven positions with p > 0.1 are described as having no detected condition dependence. An
  equivalence test (two one-sided, 90% intervals of each condition's calibrated deviation from the
  pooled mean within +-0.1 on the scan's scale, where stop ~ -0.5 and wild type ~ 0) is run on all 70 testable
  positions; the margin is a choice, stated. Predicted: 0-7 positions equivalent at +-0.1, fewer at +-0.05.
  The replicate SDs come from three replicates, so if anything they make equivalence too easy.
- The yeast sequence (the scan's wild type) is scored by ESM-2 8M and 650M: masked-marginal log-ratio for
  every single substitution, Spearman with fitness per condition. Predicted: 8M 0.15-0.35, 650M 0.35-0.55 under
  DMSO; the ordering of conditions not predicted. Yeast ubiquitin is itself in ESM-2's training data, so this
  is an alignment check, not an independent test.
