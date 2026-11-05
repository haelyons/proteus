# Measurement #6: three shuffled-weight controls at 8M (written 2026-11-05, before the atlases exist)

Atlases from job `batch_20261105` C: pretrained 8M and three controls whose weights are permuted within every
tensor (`--control shuffled --control-seed 1/2/3`; stream, hashes and plan identical). The ruler is not used
here (its rule awaits sign-off); `scripts/controls_summary.py` computes descriptive quantities per layer,
positive pole, for each atlas, then the spread across the three control seeds.

| quantity (median over the 1,280 units of a layer) | pretrained | shuffled (each seed) |
|---|---|---|
| effective number of residues in the top-64 | layer 0: 1.5-4; layers 3-5: 3-8 | 1-3 at every layer |
| effective number of residues in the peak reservoir sample (>= q99.999, up to 256) | 2-8 | 1-4 |
| units whose top-64 come from <= 3 proteins | 5-30% | 30-70% |
| units whose top-64 are one residue (effective number < 1.5) | layer 0: 20-60%, later < 20% | 40-90% |

- Seed-to-seed spread: the standard deviation across the three controls of each per-layer quantity is < 10%
  of the pretrained-minus-control difference. If it is not, a single control seed (InterPLM's design) cannot
  separate trained from untrained on that quantity.
- Why the controls should look "cleaner" on residues: permuting weights keeps a distinct embedding per amino
  acid and mixes it randomly, so units of an untrained network respond mostly to residue identity, the
  easiest thing to call "specific". Residue concentration alone does not separate trained from untrained in
  the direction one might hope.
