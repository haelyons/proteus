# autolabel: predictions

Registered 2026-11-08 by Claude (Fable 5.1), before reading the essay, janus's README, any janus tool, or the context notes. Sources at the time of writing: `autolabel/BRIEF.md` only, plus prior knowledge of the field up to mid-2026 (Bills et al. 2023 explain/simulate/score; Transluce; EleutherAI delphi detection/fuzzing scoring; Paulo et al. 2024; nnsight/NDIF; Olah et al. 2017–2020 on InceptionV1 and the mixed4e:55 "cat face plus car front" unit). I have not read the essay; it postdates my training.

Each entry: what I expect, why, and what would show me wrong. These are to be scored, not defended.

## Q1. His approach, precisely

**Expect.** The approach has two halves and a claim that joins them.

- *Responds to.* Collect activations over a large corpus and look at more than the top-k list: the whole distribution, including the middle and the near misses, plus feature visualisation. The famous polysemantic readings came from top-k lists, which show several modes and hide the one feature that links them.
- *Wired.* Read the unit's input weights (which earlier units excite and inhibit it) and its output weights (which later units read it, and what it promotes). Compare the unit to its closest peers. A single feature explains both the activation pattern and the wiring; a mixture does not.
- *Claim.* Units in trained networks (at least InceptionV1, probably also transformer MLP units) are individually legible when read this way: one feature per unit, with the apparent extra modes being that one feature seen from an unexpected angle. "Clean" means the network's own units are its building blocks, so one does not need a dictionary (SAE) to find concepts.
- *Measures.* I expect a predictive test: a label is accepted if it predicts held-out activations, including which near-miss images fail to fire the unit, and if it is consistent with the weights. I expect he reports a success count over his 1000 investigations rather than a continuous metric.
- *Failure.* A unit where no single description predicts held-out activations and wiring at once; or where the wiring-based reading and the data-based reading disagree and cannot be reconciled; or where a family of peers shows the "one feature" is shared and the unit's own identity is not what the label says. If his labels did not predict held-out activations better than the top-k label, the approach fails.
- *Which steps address which half.* Dataset examples, distribution, feature vis, held-out prediction: responds to. Input weights, output weights, peer comparison, downstream readers: wired. The join is the step where one proposes a single feature and checks it against both.

**Why.** The brief says the earlier ruler measured what a unit responds to and his method is about how it is wired, and that the decisive clue in Helios's runs was in a reader. The brief also says his own blind step filed his famous unit with planted mixtures, which implies his method has a blind step with a mixture/single decision.

**Wrong if.** He does not use weights at all and the whole method is data-side (richer stimuli, contrast sets); or the claim is about SAE latents rather than units; or there is no held-out test.

## Q2. Current pipelines and where compute goes

**Expect.**

- *Field auto-interp (Bills et al., delphi, Transluce).* One forward pass over a corpus with hooks saves activations for every unit at once; a per-unit top-k reduce; one LLM call per unit to write a label from top examples (plus a few random/quantile examples); then scoring by simulation (an LLM predicts per-token activations, many calls) or by detection/fuzzing (cheaper classification calls). Compute: collection is amortised across units and cheap per unit; labelling is one call per unit; scoring dominates LLM cost. Everything after collection is embarrassingly parallel across units because no unit's label depends on another's.
- *nnsight.* Instrumentation (tracing, intervention graph), batched invocations, remote execution on NDIF for large models. It does not itself label; it is how one collects and intervenes. Compute goes to forward passes with saved tensors; the batching is across prompts, not across units.
- *Goodfire's 1000 investigations.* Semi-automated agentic loop per neuron: top examples, feature vis, weight reads, hypothesis, probe with new stimuli, revise. Compute: many LLM calls per unit in a sequential loop plus small forward passes for probes. This is the expensive part and it is sequential within a unit (each step depends on the last) but parallel across units.
- *Sequential and why.* (a) Within-unit iterative reading: the next probe depends on the current hypothesis. (b) Family comparison: needs every unit's activations before any family can be built (a barrier, then parallel). (c) Downstream readers: reading unit u via later units needs those later units read first, a layer-order dependency. (d) Top-k over a corpus is a streaming reduce: parallel over shards, with a merge.

**Why.** Data dependencies are what make steps sequential; the only true dependencies I can see are within-unit loops and cross-unit comparisons.

**Wrong if.** Goodfire's loop is not agentic but a fixed battery run in one batch (then it is embarrassingly parallel); or the field's scoring step is cheaper than collection at scale; or nnsight has unit-level batching I do not know about.

## Q3. What janus established, round by round

**Expect.** Three phases.

- *Rounds 1 to roughly 10 (mixed4e:55 from top-k data).* Every arm reads "small furry mammal faces" and misses cars and pool tables. This is first a finding about the data (top-k lists hide the linking feature) and second about the reader (prompts that invite probing lead the reader to the field's 2017 two-job reading). It says little about the unit.
- *Rounds roughly 10 to 32 (more views, planted units).* Adding views (near misses, peers, circuit weights, feature vis, pose or range profiles, curves) changes the reading in some arms, and planted text units over Pile-10k test whether the instrument can call one-job, two-job and noise units. I expect: planted one-job units are read correctly; planted two-job mixtures are sometimes read as one feature when the reader is told units are clean (a reader finding: cleanness assumption leaks); noise units are over-read. I expect the peer/family view to be the one that most moved readings, and the circuit view to be hard for the reader to use from raw weights. Some rounds will have been voided for leakage or design flaws.
- *Rounds 33 to 36 (OLMo-2 text units, one meaning many forms, growth across checkpoints).* The reader can group surface forms when shown them side by side but does not propose the shared meaning from the top-k list alone; neuron growth across checkpoints shows the feature sharpening, which is a unit finding, not a reader finding.
- *Overall.* No arm recovered his reading of mixed4e:55 from recorded data alone; when his blind step was applied to his unit it filed it with mixtures (brief). So janus established that the reader is not the bottleneck as much as the view is, and that whether mixed4e:55 is one feature remains unresolved by janus.

**Why.** The brief's early signal, plus the general pattern that LLM readers reproduce the field's consensus when given freedom.

**Wrong if.** Some arm did recover a single-feature reading of mixed4e:55; or planted mixtures were called reliably in every arm; or the OLMo rounds showed the reader fails even with forms side by side.

## Q4. Recovering his labels as a calibration

**Expect.** Take the units he studied in InceptionV1 (and whichever transformer he covered) with his published labels as a reference set; mix in planted controls (one-job, two-job, noise) and shuffled-weight units; run the harness blind with predictions committed; score three things: agreement with his label, held-out predictive accuracy of the harness's label, and the mixture/single call on controls. A harness reads his way, rather than assuming cleanness, if it (a) calls planted mixtures mixtures and planted one-job units single at the right rates, (b) recovers his labels where his labels predict held-out activations, and (c) disagrees with him on units where held-out data disagree with his label. Agreement with him alone is not enough: a harness primed to find one feature will agree with him on everything, including the mixtures. The shuffled-weights control is the check that wiring-based labels carry information.

**Why.** The brief: test the instrument on units with known jobs first; write the decision rule before any number; the blind step filed his unit with mixtures, so the calibration must be allowed to disagree with him.

**Wrong if.** His 1000 investigations are not on a model we can run (then no calibration on his model is possible); or his labels come with held-out tests we can replay directly (then calibration is simpler than I describe).

## Q5. Where a better harness or kernel matters most

**Expect.** The checks, then collection; the reading loop only gets faster.

- *Collection.* An exact atlas (proteus: 169k units in an hour on one A100) changes what is possible because the near-miss and family checks need more than top-k: they need the full distribution or at least a large exact reservoir per unit. Approximate top-k with ties (bf16) makes near misses ill-defined. So the kernel enables the checks rather than just speeding collection.
- *Reading loop.* Batching LLM calls across units makes it cheaper and faster but does not change what can be read. Restructuring the loop to propose several hypotheses in parallel and test them with one batched forward pass would turn a sequential loop into a parallel one, which changes the cost model more than the capability.
- *Checks.* Families by co-activation are one Gram matrix over stored activations, cheap once the atlas exists and impossible without it. Downstream readers need output weights and the next layer's labels; the layer dependency is real but the per-layer work is parallel. Held-out prediction is a batched forward pass per unit over synthesised or held-out stimuli; cheap and parallel. Near misses need the distribution.
- *Most.* The single biggest lever is collecting enough per unit in one pass that the checks become table lookups: top-k plus quantile samples plus co-activation sums. That is a harness and kernel problem, and it is where parallelism changes what is possible.

**Why.** The brief's own reproduction found the decisive clue in a reader and the best signal in peer comparison; both need all-unit data, which is the collection problem at scale.

**Wrong if.** Measurement shows the reading loop dominates cost by an order of magnitude and the checks are cheap even without a kernel; or his method needs per-unit probing stimuli that cannot be batched.

## General uncertainties

- I do not know what model(s) the 1000 investigations cover.
- I do not know whether his method uses weights at all; the brief implies it.
- I do not know how janus scored readings (CONTRACT.md will say).
- I have not measured any cost centre; Q2 and Q5 are reasoning from data dependencies, not timings.

---

# Scored 2026-11-08, after the reads (sealed text above unchanged)

Evidence in `OBSERVATIONS.md` and `ANSWERS.md`; essay lines as there.

## Q1: partly held
- Held: two halves (responds-to, wired) and the join (Essay L55, L87); the claim that one feature explains apparent
  mixtures; the use of weights and readers; the one-feature question as the first step (L85, L111).
- Failed: "I expect a predictive test: a label is accepted if it predicts held-out activations". He says the
  opposite: prediction and simulation scores are tools he trusts "only up to a point" and a noun label "could do well
  on many tests ... while completely missing what it really is" (L409, L573). Acceptance is by specific criticism and
  convergence of lenses (L55), not a threshold.
- Failed: "a success count over his 1000 investigations". The gallery carries two 1-10 scores per neuron and the
  essay reports fractions, but he calls any single score "a broad pointer" (L43, L645).
- Missed: the grid-reading claim (the V was visible in the same grid), steering as a check, the sign-regime
  artefact, and the crossword (understanding compounds across neurons, L47-57), which is the step that makes his
  approach non-parallel.
- Failure criteria: I predicted "no single description predicts held-out activations"; his are a second circuit of
  the right shape, an inexplicable group, an unrelated bundle that still drives the neuron, distant concepts that
  fire, a branch without one origin (L33, L407, L645, L655, L827).

## Q2: mostly held
- Held: one amortised collection pass; one call per unit for the label; scoring dominates language-model cost
  (Paulo's table: simulation 5-30x detection; Transluce 2,891 GPU hours); nnsight batches inputs not units; Goodfire's
  loop is agentic; sequential parts are the within-unit loop, the family barrier and the reader/layer chain.
- Wrong in emphasis: I did not predict that the field's pipelines keep only top exemplars (Transluce) and that under
  those instruments neurons score well below SAE latents (Paulo Table A10), which is the data the essay argues
  against. I also did not predict that Goodfire's per-neuron procedure is not public (notes 403, paper unpublished).
- Missed: that part of the within-unit loop is not a loop (janus R33: seeing the experiments ~ designing them), and
  that the network-side checks (bundles, ablations) are absent from every field pipeline.

## Q3: partly held
- Held: no arm recovered a single-feature reading of 4e:55; the blind step filed it with mixtures (R17c); whether 55
  is one feature stays unresolved, and janus added evidence against the V (pool tables yes, flames no, round cats
  yes, R18-19); the circuit view by contribution gave Zoom In's reading (R6).
- Failed: "planted two-job mixtures are sometimes read as one feature when the reader is told units are clean". The
  clean-primed arms C2 and C4 did not unify packed controls (R1); only A0 and C0 did, 2 of 3, on one card.
- Failed: "the peer/family view most moved readings". Views that moved the reader's reading were crops, the range,
  the spared view and the invariance profile (R21-24); peers moved the measurement, not the reader (R27-29), and the
  reader named no side until a table listed it.
- Failed: the OLMo rounds. R34 was a decoding experiment, not a reader test; R36 found early cores unrelated to the
  final contexts rather than "the feature sharpening" (narrowing held, 83%, but cores not kept).
- Missed: R35 exists (agent labels from feature vis plus top-5; human half blank) and R32-33's basis results
  (random directions readable).

## Q4: held in outline, wrong about the leak
- Held: controls with known jobs, disagreement with him allowed, agreement alone insufficient, shuffled-weight
  control.
- Wrong: I framed the risk as "a harness primed to find one feature". janus shows the prompt is not the leak;
  confirmation in probing, the cheap "unrelated" exit, typical-vs-distinctive, and overclaiming prose are (R2, R9,
  R11, R29-30).
- Missed: the richest calibration set is the Qwen gallery with its test rows (open weights), and his InceptionV1
  labels already failed within class in R30A. Also the circularity of his test rows.

## Q5: held in outline
- Held: collection with range and co-activation changes what is possible; the reading loop mostly gets faster;
  checks are where the structure is.
- Added by the evidence: the loop splits into a standing battery (parallel) and a residue (sequential), per R33 and
  R30B; the reader/crossword chain is the one truly sequential part and no pipeline has built it; the network-side
  checks' cost at scale is unmeasured, and so is the cost of one of his neuron papers.
- Not measured by me: any cost centre. The timings cited are janus's, proteus's, Transluce's and Paulo's.

## General uncertainties, resolved or not
- Models covered: Qwen3-32B (dense) and a MoE (presumably Qwen3.6-35B-A3B) in the gallery; InceptionV1, three
  ConvNeXts, GPT-2 XL, Kimi K3 in the text. Resolved.
- Does the method use weights: yes (L83, L87, L407). Resolved.
- How janus scored: CONTRACT.md; mechanical class and held-out scoring, later Spearman over a fixed pool and
  within-class d against the layer. Resolved.
- Cost centres: still unmeasured.
