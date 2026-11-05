# Ruler v1 plan: what each item needs, where I disagree, what it costs (2026-11-04)

Supersedes the scoring parts of `ruler_v0_predictions.md` (the single-residue vs window-null discussion);
its descriptor alphabet, redundancy rule and persistence criterion carry over. Nothing here has been
computed on real units. Costs: GPU at Lambda prices (A10 $1.29/h, A100 $1.99/h); engineering in my days.

## 1. Validate the ruler on columns with known jobs (first)

Agree, and it gates everything else. Design:
- Synthetic columns, scored per token from the descriptors and sequence, then pushed through the
  same atlas code as model columns: helix indicator, burial count, tryptophan, a two-job mixture
  (helix in one half of UniRef50 clusters, strand in the other), pure noise, and a sequence-only job
  (hydrophobic moment of the +-3 window), which tests item 2's logic.
- "Realistic noise": I would not pick one level. Each synthetic score is rank-mapped onto the marginal
  distribution of a real 8M unit (so tails look like real tails) and mixed with noise at an SNR ladder
  (e.g. 8 levels). The ruler's output is then a detection curve per job type; "calls correctly" means
  above a stated SNR, with the false-call rate at SNR 0 (pure noise) as the floor.
- A planted family: 20 noisy copies of a helix detector, each with one distinct extra axis (N-cap,
  buried face, Ser-rich, near a disulfide). This validates item 3's contrastive step the same way.
- Positive control on real data: InterPLM's 8M SAEs (all six layers are on HF); their concept labels
  are in a separate data release I have not fetched yet.

Cost: ~1.5 days engineering; GPU ~$1 (8M-scale column counts over Swiss-Prot on an A10). Needs the
whole-build descriptors (running now).

## 2. A trivial-sequence baseline: yes, but it does not retire the null

Agree that the comparison that decides "learned" is against a model that sees only local sequence.
Two corrections:
- Calibration still needs the composition-matched null. The baseline answers "is this more than local
  statistics?"; the null answers "is this score more than sample size and composition?". Every column,
  baseline columns included, gets the same calibrated score, and the comparison is between calibrated
  scores. Without it a tryptophan-heavy baseline column gets structural coherence from composition.
- Random projections of +-3 one-hot windows are a floor, not a baseline: sums of many terms give
  near-Gaussian columns whose tails mix many residue combinations. The baseline that can embarrass
  ESM is the best a +-3 model can do: (a) a small CNN masked LM with receptive field 7, trained on
  the same Swiss-Prot split, its hidden units as columns; (b) per descriptor, the Bayes-optimal local
  predictor (logistic regression on the +-3 one-hot window), which is the ceiling of the local
  explanation for that descriptor. Report ESM coherence beyond (b) for the unit's own top descriptor,
  and the floor (random projections) for scale. Add +-10 as a second window so the "local" curve is
  visible.

Cost: ~0.5 day; GPU ~$1 (training the +-3 CNN, ~30-60 min on an A10); predictors on CPU.

## 3. Families: agree, but our Gram cannot be the "contribution on data" side

Agree with family-relative scoring and with reporting weight-based and data-based families separately.
The correction: our Gram is the uncentered second moment of the pre-GELU activation h = W_in LN(x) + b
on the hashed subsample, i.e. W_in Sigma_x W_in^T plus bias terms. It is input-weight similarity
measured in the metric of the input covariance, dominated by the bulk below zero where GELU passes
almost nothing. Gram-based and input-weight rankings will differ only by the anisotropy of Sigma_x.
They will not reproduce the weight-vs-contribution split from your image work.

The data-side relation we already have: the reservoirs are bottom-k sketches under one hash shared by
every unit and every layer, so the Jaccard overlap of any two units' tail sets (q99.99 and up) is
estimable from the stored samples alone, within and across layers. Standard error ~sqrt(J(1-J)/k):
+-0.06 at k = 64, +-0.03 at k = 256.

Families therefore come three ways:
- input weights (rows of W_in);
- output weights (columns of W_out);
- tail co-firing (sketch Jaccard).

The Gram is the fourth, bulk view, and we keep it. Contrastive coherence then asks which descriptor
separates a unit's tail samples from its nearest family members' tail samples. With 64 samples and
~15 descriptor states, choosing the best-separating axis selects on noise: the null is a permutation
of unit labels within the family, repeated with the same selection step, and BH across units. K_r = 256
helps here more than anywhere.

Cost: ~1 day; GPU none for weights and sketches; minutes for the Gram on the filesystem.

## 4. Downstream: agree, with two limits on the weight-only version

- ESM-2's head is not an unembedding: final LayerNorm, then dense, GELU, LayerNorm, decoder (tied to the
  embeddings). Projecting W_out[:, u] "through the unembedding" needs a linearisation (LayerNorm scale
  and GELU slope frozen at their typical values) or a measurement on data (direct-logit attribution
  through the real head at the unit's top tokens). I would do both and report where they disagree.
- Direct effects are small for early layers; their effect on the loss is mostly through later layers.
  The direct-logit view is the job "in the only sense the loss cares about" only for late layers.
- Readers by weight: fold each reader's LayerNorm gain into its input row and centre the writer's output
  column (LayerNorm removes the mean). Even so the dot product sees only the same-position path;
  readers at other positions (through attention) are invisible to it. Validate on a sample by exact
  ablation (zero the unit, measure later units) at 8M.

Cost: ~0.5 day; GPU ~$0.3 (8M ablation sample).

## 5. The whole chain at 8M first: agree

Columns, all through one atlas build:
- 7,680 units;
- 61,440 InterPLM latents (6 layers x 10,240);
- the shuffled-weights and random-init controls;
- the baselines (item 2);
- the synthetic columns (item 1).

Settings: K_r = 256, the q99.999 reservoir as the peak, families, persistence, held-out.

Held-out "strict" subset: clusters are re-formed every UniProt release, and our UniRef50 mapping is the
current release and Swiss-Prot members only, so "no pre-2021 member in its UniRef50 cluster" cannot be
read off it. Defined instead as: no hit at >= 50% identity (MMseqs2) in UniRef50 2021_04 (the release
ESM-2 trained on), with a 30% variant as an even stricter set. Prediction: small, 200-1,000 of 10,010
proteins, so that test will be low-powered.

Cost: GPU ~$2-3 (8M passes for units, two controls, latents, baselines, synthetic columns on an A10);
UniRef50 2021_04 download + MMseqs2 ~$2-3 on a cheap instance; engineering ~1 day (column sources
for SAE latents and baselines in the atlas driver).

## 6. Source-position attribution: agree; exact masking at 8M first

For each of a few hundred units' q99.999 samples, mask each position j of the protein (one forward
per (protein, j), which gives the effect on every unit at every target position at once) and record
the change in the unit's activation at its target position i. Map |delta| over j against sequence
separation |i - j| and spatial distance (C-beta, now in the descriptors). Then check whether
gradient x input agrees with masking on the same samples; only if it does is it the scalable proxy for
650M.

Cost: ~1 day; GPU ~$0.5 at 8M (~5,000 proteins x ~350 positions of the 8M model, ~10-20 min on an A10);
the same at 650M would be ~2-3 h.

## 7. Decision rule (v1, revised 2026-11-05 after the assessment; numbers frozen before the first real-column score)

What the ruler can say: whether a column's samples are associated with our descriptor vocabulary beyond
composition and sample size. It cannot say what a column computes, and it cannot tell "no job" from "a
job outside the vocabulary". Revisions from the draft: persistence is descriptive, not a requirement
(a narrow detector whose meaning changes at weaker activation is still associated); "no association
found by this procedure" is a category of its own, never read as "unrelated jobs"; the negative pole is
the unit's off-state (GELU writes ~0 at its extremes, `assessment_20261104.md` section 5), so
structural and functional claims use the positive pole, and the negative pole is reported only as where
the unit is silent.

Blinding: every column (unit, latent, baseline, control, synthetic) gets an opaque id; the key file's
sha256 is committed before scoring and the key after the calls are frozen.

Calls per column (positive pole; samples de-duplicated by UniRef50 cluster; BH over columns x descriptors):
- "structural association": at least one structural descriptor (state, burial, backbone region)
  enriched at the peak (q99.999 reservoir) beyond the composition-matched null. Sub-label "one mode" or
  "several modes": several enriched states of one descriptor (e.g. helix and strand) are reported as
  several modes, provisional until the family check; they are never called several jobs on their own
  (one meaning in two contexts gives the same pattern).
- "composition association": residue identity enriched against background, no structural excess.
- "context association": both, i.e. a residue response that depends on structure.
- "no association found by this procedure".
- Persistence class (descriptive): mode present at q99 and q90 / q99 only / peak only (half-effect
  criterion with 95% interval).

Gate (ruler validity) on synthetic columns, else no call on real columns, at the stated SNR or above:
- single-job columns: "structural association, one mode" >= 90%, the right descriptor;
- tryptophan: "composition association" >= 90%;
- residue-in-context column: "context association" >= 80%;
- planted two-job mixture and planted one-meaning-two-contexts column: both "several modes" >= 80%
  (the ruler is expected not to separate them; that is the family check's job);
- narrow detector with a different meaning at weaker activation: associated at the peak >= 90%, and
  persistence "peak only" for its structural mode >= 80%;
- pure noise: "no association found" >= 95%.

Population calls (8M, per layer and pooled; f = fraction of columns with an association beyond the +-3
Bayes-optimal local predictor for the same descriptor; S = shuffled-weights units, three seeds, as the
false-call floor):
- "units associated": f_units - f_S >= 0.30 and f_units >= 0.75 f_latents.
- "latents associated, units not": f_latents - f_S >= 0.30 and f_units < 0.5 f_latents.
- "both": both >= 0.30 above the floor and f_units / f_latents in [0.75, 1.33].
- "neither": both < 0.15 above the floor.
- Otherwise "inconclusive", stating which criterion failed. Held-out: on the strict held-out subset
  (no hit >= 50% identity in UniRef50 2021_04), the predicted enrichment holds in sign for >= 70% of the
  associated columns, or the call is reported as not generalising.

Numbers are placeholders until accepted; once accepted they are frozen in a commit before the first
real-column score.

## Housekeeping

- Checksums: the descriptor job writes a sha256 manifest of every run file on us-east-1; asia-south-1
  (science pass) and us-west-1 (impl_ab) need a few instance-minutes each when capacity exists (~$0.3).
- K_r = 256 in the next build: reservoir state 4x (650M: ~10 GB on an A100 40 GB, fits); more
  reservoir candidates; my estimate +5-10% pass time. For 8M it is negligible.
- Fragment-geometry coherence (+-5 phi/psi fragments): dated 2026-11-12, after the synthetic validation.
- The Gram is kept and consumed (item 3, bulk view).

## Order and total

Descriptors (running) -> synthetic columns + baselines + 8M atlas build (units, controls, latents,
baselines, synthetic, K_r = 256) -> ruler validation gate -> families, downstream, persistence,
held-out -> source attribution sample -> decision -> 650M. GPU for the whole 8M chain: ~$8-12.
Engineering: ~5-6 days, which lands before 21-22 November if nothing large breaks.
