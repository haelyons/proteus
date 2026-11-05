# Round 2 plan: real-unit cases, baseline repair, pair balance (frozen 2026-11-05, before real-unit calls exist)

Ruler v1.2 stays frozen (atlas/ruler.py @ aab0c82). The challenges found no failure under the tested
conditions; the high-signal error bound is loose (upper 12%) and the four sets shared their noise. Until the
repaired baseline passes, every structural or context call carries "sequence-only not excluded".

## 1. Case selection (8M pretrained atlas, positive pole), fixed before any real-unit call is seen

Calls come from `scripts/score_real_units.py` over all 15,360 columns (6 layers x 1,280 units x positive pole),
with descriptors for the whole stream; BH across the atlas's columns.

- Eligibility: peak `n_clusters >= 40`.
- **Random picks:** structural x2, composition x1, context x1, "no association found" x1. For each category,
  the eligible columns are ordered by a permutation from `torch.Generator().manual_seed(20261105)`, built
  over the list of (layer, unit) sorted ascending. The first eligible column is taken; two for structural.
- **Fallback:** if a pick fails a practical requirement found later, take the next column in the same
  permutation and record why. The only such requirement is fewer than 20 fresh positions above the unit's
  discovery q99.9. Categories with no eligible column are recorded as empty.
- **Deliberate picks** (never estimates of how often units are understandable):
  1. the valine pair (layer 1, units 86 and 1200);
  2. a "not detected" unit chosen for structured wiring: among eligible no-association columns, the one
     whose output weight has the largest cosine with any next-layer unit's input weight (a single strong
     reader); ties go to the lower layer, then the lower unit;
  3. a 650M ubiquitin-enriched unit: among 650M columns with >= half of their top-64 in ubiquitin copies, the
     one with the largest share; ties go to the lower layer, then the lower unit.
- Proteins:
  - **discovery:** non-held-out proteins (the stream the atlas came from);
  - **fresh:** date-held-out proteins excluding the 400 used in the first pair experiment (seed 0), with 600
    drawn by seed 1;
  - **strict:** the 111 held-out proteins with descriptors and no hit >= 30% identity to UniRef50 2021_04
    representatives, reported separately where support allows.

Per case, the account is:
1. the proposed computation, written after inspecting discovery material only (examples, weaker bins,
   ruler scores, downstream readers, intervention smoke on discovery proteins), with that material recorded;
2. predictions for fresh inputs and interventions, frozen;
3. the confirmatory run;
4. what is supported or contradicted, and what is explained beyond naming what activates the unit.

Unsuccessful explanations are kept.

Interventions:
- post-GELU silencing (and scaling) at a position;
- controls are other units of the same layer, matched in the run itself on the magnitude of the perturbation
  actually applied, i.e. the norm of the change in the write vector, |dGELU| x |w_out|;
- readouts are signed: change in log p(true residue) at masked positions (positive = helps). "Moves
  predictions" (KL, |change|) is reported separately from "helps prediction" (signed change for the correct
  residue).

## 2. Baseline v1 (repair of v0), defined before it is run on anything

**Target.** For a column's peak sample (bin 5, one token per UniRef50 cluster) and the ruler's primary
descriptor x, define

    p_W(x) = E_bg[ r(w) 1{x} ] / E_bg[ r(w) ]

where r(w) = P(window w | peak) / P(window w | background) and w is the +-5 residue window. It is the
descriptor distribution a selection that depends only on the window, with the peak's own window preferences,
would produce. Same level: r is fitted on peak tokens, not on lower bins. v0 fitted bins 3-4 and tested the
peak; a sequence-driven column can prefer different windows at different levels (a motif dominates the peak,
while weaker sequence features and noise share the lower bins). So v0's mismatch was in the target, not only
in the model.

**Estimation.**
- **Folds:** 5 folds by UniRef50 cluster (hash of the cluster id).
- **Fitting:** for fold f, r^f is fitted on peak tokens and 20,000 background tokens from the other folds'
  clusters, with a classifier that can represent residue conjunctions: an MLP, one-hot window (11 x 21) ->
  32 ReLU -> 1, weight decay 1e-3, Adam (lr 1e-2), 300 full-batch steps, positives weighted to balance.
- **Expectation:** fold f's expected distribution comes from background tokens of fold f's clusters only,
  reweighted by r^f.
- **Test:** observed counts of the peak tokens, pooled over folds, against the sum over folds of
  n_f x p_W^f; Monte Carlo G with 2,000 draws, sampled per fold.

**Support limits (stated, not tuned):**
- "not estimable" when fewer than 40 peak tokens with a defined descriptor, or when the smallest fold's
  effective sample size of the reweighting is below 100;
- with K_r = 64 (the synthetic sets) a fold trains on ~45 positives; with K_r = 256 (the 8M atlas) on ~180;
- a conjunction the MLP cannot learn from that many positives stays unexplained. That is a limit of this
  estimator, and so the wording is fixed: "unexplained by this fitted +-5 window model".

**Pass criteria** (structural / context calls of challenge set A as scored, K_r 64, and of set A regenerated at
K_r 256 with the same noise):
- sequence-only columns (motif_nxst, motif_gxgxxg, hydro_win): "unexplained" in <= 1 per set;
- structural columns (helix, buried, ordered, two_job, conj_exposed_strand, narrow, lys_helix): "unexplained"
  in >= 80% of estimable calls (power);
- report "not estimable" counts.

**Fresh challenges** (set E, K_r 256, new noise seed; expectations written before generation):
- new sequence-only types: hp_i34 (hydrophobic at i, i+3, i+4), kxxxe (K at i, E at i+4), pxxp (P at i
  and i+3);
- new structural types: strand_buried (strand AND buried tercile), helix_exposed (helix AND exposed tercile);
- the same pass criteria.

If v1 fails, the failure is reported and "sequence-only not excluded" stays on every structural call.

## 3. The pair: graded interventions and natural variation

- **Graded** (what output changes each contribution can produce): scale 86's and 1200's post-GELU outputs
  independently over {0, 0.5, 1, 1.5, 2} at fresh "both" positions. Report the readout surface and
  whether joint changes equal the vector sum of the individual changes.
- **Readout, defined now:**
  - polar-minus-hydrophobic index at a position = mean log p over {E, Q, K, G} minus mean log p over {I, L, F},
    from the first experiment's largest movers; it is a hypothesis being tested, not a label;
  - signed change in log p(true residue) at masked positions;
  - KL, reported as "moves predictions", never as "helps".
- **Natural variation:**
  - on discovery proteins, fit the relationship between the two units' pre-GELU activations where either is
    above its q99 (linear fit of h_1200 on h_86);
  - define departures as residuals beyond the discovery residual's 10th and 90th percentiles;
  - characterise their contexts on discovery proteins (residue identity at -3..+3, state5, burial3);
  - write the proposed context and its predictions, have them challenged by an independent reviewer (an
    agent given only the proposal and the record of what was inspected, asked for an alternative explanation
    and an observation that distinguishes them), then freeze the test;
  - test on fresh proteins: (a) do departures recur in the proposed context? (b) at departure positions,
    replace the pair's output by what the fitted relationship implies (removing the natural variation) and
    measure the signed change in log p(true) at masked positions within +-3, in the predicted context
    versus elsewhere, against controls matched on the perturbation norm applied in that masked run;
  - compare joint effects with the vector sum of individual effects, so that ordinary cancellation is
    not taken as purpose.

## 4. 650M ubiquitin-enriched unit

Question: does its role extend beyond recognising sequences close to ubiquitin? Family recognition is itself a
legitimate finding.
- **Discovery:** its top-64 and reservoir samples (exported from atlas #2).
- **Fresh probes, written before running:** ubiquitin, close relatives (e.g. NEDD8, ~58% identical),
  distant beta-grasp folds (e.g. SUMO1, ~18%), ubiquitin-like domains embedded in other proteins,
  composition-preserving shuffles, and point mutants at the positions where it fires.

This proceeds regardless of 8M's weak DMS correlation.
