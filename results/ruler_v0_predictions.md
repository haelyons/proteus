# Ruler v0: definitions, positions and predictions (written 2026-11-04, before any coherence number)

The ruler decides the hypothesis, so it is fixed here before it is computed. Evidence used so far: a
2,000-protein random Swiss-Prot sample of AlphaFold DB models (`scripts/afdb_descriptors.py`), the
shuffled-weights control's definition (`atlas/esm.py`), and the 650M atlas's bin structure.

## Samples and descriptors

- Units: positive pole for structural claims (GELU passes little of the negative pole). Levels: top-64,
  and reservoir bins q99.99-q99.999, q99.9-q99.99, q99-q99.9, q90-q99 (64 uniform samples each).
- State per residue: helix, strand, loop (pLDDT >= 70), disordered (pLDDT < 50), uncertain (50-70).
  Evidence: pLDDT < 70 is 16.3% of residues, < 50 is 10.2% (sample of 698k residues). Whole model-organism
  proteomes put 30-40% of residues below 70, and only part of that is disorder; Swiss-Prot is less
  disordered (bacteria-heavy, well studied). Among residues < 70, pydssp calls 18% helix: AlphaFold
  draws helices in low-confidence regions. So the 50-70 band (6.1%) is mixed, and I keep it separate
  rather than calling it disorder. The 4-state version (< 70 = disordered) is reported alongside.
- Environment (only where pLDDT >= 70): burial terciles (C-beta count within 10 A). Contacts within 8 A
  correlate with burial almost by construction and are dropped from the score.
- Local backbone (pLDDT >= 70): Ramachandran region of the residue (alpha-R, beta, alpha-L, other); the
  +-5 fragment classes come later.
- Amino-acid identity is its own axis, scored against the background composition. A tryptophan detector
  has a job; it is a residue-identity job, and it is reported as one, not as structure.

## Score: mutual information against a composition-matched null

For each (unit, level) and descriptor X: the bin's samples S (after redundancy control, below) and a null
set N of the same size drawn from all non-held-out Swiss-Prot residues with exactly S's amino-acid
composition (and a valid descriptor). Score = MI between membership (S vs N) and X, i.e. the
Jensen-Shannon divergence of P(X | S) and P(X | N), minus its expectation when S is itself drawn from
the null (resampled), with a resampling p-value. It decomposes per descriptor as you say.

Where I disagree: per-descriptor MI cannot tell "several jobs" from "one job with correlated
descriptors". Helices are often buried on one face; disordered residues are exposed and loop by
construction in AlphaFold models. A disorder detector will light up loop, exposure and low pLDDT at
once. So v0 reports, next to each marginal MI, the conditional MI given state, I(bin; X | state). The
multi-job claim needs a mixture test (the samples split into subsets coherent in different
descriptors), which per-descriptor MI cannot see. I am not proposing dispersion: I have no number that
favours it.

## Redundancy

At most one sample per UniRef50 cluster per bin (the smallest hash key, so the choice stays uniform and
reproducible); `n_clusters` reported with every score; bins with < 10 clusters are unscorable, not
scored low. Predictions for the 650M atlas (pole 0): top-64 lists median 10-25 clusters, 25-50% below
10; q99.999 bins 3-15% below 10; q99.99 1-5%; q99.9 and lower < 1%. The top-64 is the level most
inflated by redundancy (proteins arrive whole and tails cluster within proteins; we measured that for
the seeds), which matters because it is the level the persistence test starts from.

## Controls: predictions

The shuffled control permutes values within every tensor, embeddings included, so every amino acid
still has a distinct (random) embedding and rotary positions are untouched. Prediction:
- MI(bin; amino acid) in the shuffled model is large in early layers, comparable to the trained model.
  It is not a sign the null is wrong; it is why the null must be composition-matched.
- Structural MI beyond composition is small but not zero. Random attention mixes neighbouring
  residues, and local sequence windows carry secondary-structure propensity, which matching one
  residue does not remove. Prediction: units with significant structural excess (p < 0.01 after BH),
  q99.99 bin: trained 40-70%, shuffled 5-15%; shuffled median excess <= 0.2x trained. If the
  shuffled fraction exceeds 25%, the null gets matched on the +-3 window composition before anything
  is said about the trained model.

## Persistence: predictions

"Persists to level L" = the descriptor state most enriched at the top-64 (largest excess, significant)
is also enriched at L with a 95% interval that excludes zero and a log-odds at least half the peak's.
Significance alone is not enough: with 64 samples a q90 bin cannot detect a modest enrichment, so
"noise below the peak" would partly be a power artefact; the half-effect criterion and its interval
are what can call a mode absent. Among units scorable at all three levels:

| layers | persists to q99 | persists to q90 |
|---|---|---|
| 0-5 | 50-70% | 30-50% |
| 6-24 | 35-55% | 15-30% |
| 25-32 | 25-45% | 10-25% |
| all | 35-55% | 15-30% |

Early layers persist more because their jobs are residue identity and local context, which hold
across levels.

## Data quality

- Sequence check against 2026_03: 3 of 2,000 proteins mismatch (0.15%); 68 (3.4%) have no AlphaFold DB
  model: 42 viral (AFDB leaves viruses out), 8 with non-standard residues, 18 without a found reason.
  Together 3.3% of residues. The ruler therefore cannot see viral proteins; the held-out test inherits it.
- Held-out set: atlas over the 10,010 held-out proteins with the main atlas's bin edges, and their
  descriptors, are in `jobs/followup_650m`.
