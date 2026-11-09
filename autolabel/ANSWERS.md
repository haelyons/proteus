# autolabel: the five questions, discussed

Written 2026-11-08 after `OBSERVATIONS.md`. This is discussion, not a design: no harness is proposed here. Citations
as in OBSERVATIONS.md (Essay L<n> = `janus/authors/informed.prompt_sent.md`; README L<n> = `janus/README.md`).

## 0. Two places where the brief's reading does not match the text

0.1 The brief says the famous units were misread "because a list of top examples hides the one feature that links
    them". The essay says the opposite about the list: he saw the V in the same grid "I'd seen hundreds of times"
    once he asked "what one thing this might be" (L85). What hid it, on his account, was the reader's noun habit
    (L109) and feature visualisation's whiskers (L83). The wiring (flames, jets, the O pole) came second, as
    confirmation (L87). The brief's version is closer to what janus found than to what the essay claims: no reader,
    under any prompt, saw a V in ImageNetV2's top grid (README L44, L64, L190), and the essay's confirmations live in
    a reader and in renders, not in the recorded data (L149, L458). So "the list hides it" is our finding about this
    data, and "the reader's categories hide it" is his claim. They are different claims and both may be true.
0.2 "Clean" as deeper than monosemanticity is his wording (L477): parts "built to be read, and built upon, by its
    downstream readers". That reading is right. What follows from it is less often noticed: the test of a reading is
    whether downstream readers use the distinction, which is a wiring question, and the essay's own 4e:55 case
    offers no downstream test ("the car readers do get a little cat signal ... but it barely reaches the output",
    L83, is an argument, not a measurement).

## 1. What his approach is, stated precisely

1.1 Three claims, which he holds with different confidence.
    (a) A process claim, "maybe 85% sure" (On the Title): networks grow by generalisation and composition; arbitrary
        packing of unrelated things into one neuron (his sense of superposition, A Note on "Superposition") does not
        happen in the models he has looked at; what looks like it is bad tools, bad architecture, or recursive
        generalisation (L167-175, five causes).
    (b) A score claim he explicitly holds loosely: ~90/7/2-4 on Qwen3-32B poles, threshold-dependent (L41-43);
        "the part we're least sure of" (On the Title).
    (c) A method claim: ordinary science on the network itself, many lenses, conjecture and criticism, with
        understanding that compounds across neurons (L25, L47-57, L518). Not a pipeline; he says single metrics are
        the wrong place to start (L573, L645).
1.2 What he measures. Nothing numeric for 4e:55 (L85-89: the question, the inputs and readers by weight, the
    negative pole, generic renders, new noun groups). For language neurons: agent-written neuron papers with a label,
    two 1-10 scores and 7-10 blind test rows each (OBSERVATIONS B1-B2); blind maximum against a 10B-token corpus max
    (L401); attribution bundles with a blind maximum per bundle under ablation (L407); the 200-neuron test (76% new
    examples fire, 11% near misses, ~1% distant concepts; 164 won, 15 tied, 4 lost; L649-655); weak-activation
    ablation (L595); neuron ablations inside attribution graphs (L365); SAE copies (88% near-copies of single
    neurons); growth over training (Watching Neurons Develop).
1.3 What would show it failing, in his own terms. One published circuit "with the right shape" (L33). One group of
    neurons whose joint activity stays inexplicable after audit (L645). A bundle whose explanation is unrelated to
    the others yet still drives the neuron (L407). Distant-concept texts that fire (L655). A branch that does not
    trace back to one local origin and grow in order (L827). For a specific label, a specific discrepancy, such as
    the flame connection against "cats and cars" (L409). What he says does not count: a good prediction or
    simulation score for a noun label (L409, L573); any single cleanliness number (L645, L821).
1.4 Responds-to against wired, step by step.
    Responds-to (data side): the top grid and the "what one thing" question (L85, L111); the negative pole as the
    unit's opposite (L87, L241); raw ranges rather than percentiles (L43, L333); long context and decoding (L203,
    L195); steering (L51, L199); blind maximum, new examples, near misses, distant concepts (L401, L649-655).
    Wired (weight side): strongest inputs and readers by weight (L87-89); "visualize what the neuron's inputs have in
    common" (L83); clustering upstream attribution into bundles and ablating them (L407); readers and negative logit
    connections (L33 "nothing reads that combination"; L415); ablations in attribution graphs (L365); expert
    specialisation (L51).
    The join: "look upstream at how it's built and downstream at what it does, and both line up" (L55). In the
    language-model procedure the two halves are run together on every neuron; for 4e:55 the wiring half carried the
    confirmation.
1.5 What the text leaves open, and why it matters for automation.
    - No decision rule is written down for "one thing" against "two related" against "unrelated"; he says the
      number does not matter much (L41-43). An automated harness needs one, or it needs to report a distribution.
    - His tests measure the centre of a label (L403: "a blind maximum tests the neuron's center, not its breadth").
      He says so himself: near misses do not test the superposition question (L655-657). The question "one job or
      two" is answered by bundles plus judgment, not by a score.
    - The 4e:55 wiring evidence did not reproduce from the weights in janus: by-weight inputs render as faces,
      crickets, arcs and grilles, and the flames are in a reader, mixed5a:94 (README L140-152). His figure may show
      attribution over natural images rather than weights (README L297 raises this); the essay does not say.
    - "Clean" is about readers; none of the public tests reads a neuron through what reads it. Why not? Possibly
      because a downstream test needs the next layer labelled first (his crossword, L51).

## 2. What current pipelines do and where compute goes

2.1 Four published pipelines and one instrument (OBSERVATIONS C).
    - OpenAI 2023: one activation pass; per neuron a few top excerpts to the explainer (two records at 2K context);
      simulation by a logprob trick over every token of held-out excerpts; correlation score. The essay's own figure
      for its yield is ~1.7% (L39).
    - delphi: cache over >=10M tokens in batches of 8; sample top/random/quantile examples; 40 examples of 32 tokens
      to a 70B explainer; detection or fuzzing (5 examples per call), optional simulation, surprisal, embedding,
      intervention; concurrency by a `max_concurrent` setting; vLLM offline. Costs per 100k features: ~$600 for
      detection, ~$3.6k for all-at-once simulation, ~$18.7k token-by-token.
    - Transluce: every neuron of an 8B model, both polarities, from 10-20 top exemplars, a best-of-25 distilled
      explainer and a fine-tuned simulator; $0.046 per neuron, of which simulation was 2,891 GPU hours.
    - MAIA: an agent loop per neuron with exemplars, synthesis and editing tools; 300 real units and 85 synthetic
      ones with known selectivity; predictive evaluation from the label.
    - nnsight: instrumentation. Several invokes become one batched forward with row-scoped isolation; remote runs
      ship a serialised block to NDIF with a one-hour limit and return only saved values. It batches inputs; all
      units of a layer come out of one pass anyway.
    - Goodfire's 1000 neuron papers: agentic, with the steps of L51-55 and the tests of L401-407 and L649-655; the
      gallery shows per neuron a label, scores and ~10 test activations; the notes pages are not public and the
      companion paper is not out, so the loop's cost and call count are unknown.
2.2 Where each spends compute.
    - Collection: one forward pass over the corpus per model, all units at once, plus a reduce per unit. Cost
      scales with corpus x model, not with units. What costs extra is keeping more than the top-k (quantile
      reservoirs, co-activation sums): proteus's figures put forward-only at 48.8 min and forward plus atlas at
      60.4 min for 169k units, with a dense fallback at ~28x a forward pass (OBSERVATIONS F1). The field's
      pipelines keep top-k plus some random or quantile windows and nothing else; Transluce keeps top exemplars only.
    - Labelling: one call per unit in the field (seconds, cents); 3-10 calls in a loop for MAIA, janus's experiment
      arms ($0.2-0.5, 18-28 s per session, OBSERVATIONS D2) and, presumably, Goodfire's papers (unknown).
    - Scoring: the largest language-model cost in every published pipeline (Paulo's table; Transluce's GPU hours).
      Cheaper scorers (detection, fuzzing) are 5-30x cheaper than simulation and measure something different.
    - Network-side checks: held-out activations on new stimuli (a forward per batch), attribution bundles (a backward
      pass per top example per unit: janus's bundles.py does 150 per unit), ablations per bundle (a forward per
      condition), steering (a forward per intervention). These exist in the essay's method and in janus's rounds
      16-17, and in none of the three field pipelines.
2.3 Which steps are parallel and which are not, and why.
    - Embarrassingly parallel: the corpus pass over shards (with an exact merge if the top-k threshold is monotone,
      as proteus claims); labelling across units; scoring across units and excerpts; held-out forwards across units
      if the stimuli are pooled into one batch (nnsight's invoke batching is exactly this); feature visualisation
      across units.
    - Sequential by data dependency: (a) the within-unit loop, where the next probe depends on the last answer
      (MAIA; janus E1/P1; the essay's "run specific tests"); (b) family and layer-baseline statistics, which need
      every channel on the same inputs before any unit can be placed (janus's battery, peers, pairtable all wait for
      a layer-wide cache); (c) reading a unit through its readers, which needs the next layer labelled, and his
      crossword claim that understanding compounds (L47-57), which makes the network-wide job a frontier that
      advances rather than a bag of independent tasks. (b) is a barrier, then parallel; (c) is the only true chain.
    - Why: not hardware but dependencies. The field's pipelines have none of (b) or (c) because they do not compare
      units or read through readers; that is why they are embarrassingly parallel, and also, on the essay's account,
      why they fail (L45: "couldn't run the loop of conjecture and test").
2.4 One observation about janus's round 33: when a subject that did not design the experiments was shown the same
    experiments, it did as well as the one that did (Y1 ~ P1, README L1120-1128). If that generalises, part of the
    within-unit loop is not a loop: a standing battery run for every unit in one pass gives most of the gain. Round
    30B is the counter-evidence: a standing set of eight questions found nothing on five of six random units because
    "the missing candidates dominate" (L994). So the sequential part is hypothesis generation for content the
    battery does not cover, not the testing.

## 3. What janus established, round by round, and about what

3.1 janus's question is narrower than his approach. CONTRACT.md asks whether a prompt, given plain recorded data
    from one unit, reads it his way. His approach reads a unit with its inputs, readers, negative pole, steering,
    growth and neighbours, inside a network where much is already labelled. janus gave subjects top grids, then
    added views one at a time; it never gave a subject the reader (L149: "which no subject has seen"), steering does
    not exist for InceptionV1, and the crossword context does not exist. So janus tests the data-side half of his
    method and, in rounds 6, 16 and 17, the wiring half by his own steps.
3.2 Findings about the reader.
    - Prompts and dispositions do not move the reading (R1 all six arms including the essay-informed C4; R6 N0; R8
      Q0; R11 E1). The essay's dispositions did not impose unity on packed controls either (R1, R6, R8 controls).
    - Given freedom, the reader confirms: probes stay inside animals and vehicles (R2, L64), edits test the held
      hypothesis (R9, L204), and the field's 2017 reading is rediscovered.
    - The reader needs a baseline it does not ask for: typical drops are read as evidence (R11, L265); shown the
      baseline, nothing changes (R12c).
    - The reader will not name a mirror axis unprompted even with the twin in front of it (R25-R28: 0 of 72), but
      picks the right side when a measured table lists it (R29, 4 of 4); its prose then overclaims beyond the checked
      rows (R29, R30A) and a rule does what the final reader does (L923).
    - Designing experiments adds about nothing over seeing them (R33, Y1 ~ P1).
    - On a feature-visualisation plus top-5 page, a tool-less reader labels explained units at 0.79, random neurons
      0.69, random directions 0.50 (R35, incomplete: the human half is blank).
3.3 Findings about the data and the view.
    - Top-k over a class-balanced corpus understates the cars (L75-77) and the negative grid misleads (L80-82).
    - Which ranking you show decides the reading: inputs by contribution give Zoom In's reading (R6, 3/3), inputs by
      weight surface 511 and pool tables (L133-137); crops are the big effect on shallow units (R21, L578); the
      invariance profile with a layer percentile lifts the pose unit (R21-23); direction appears only from a spared
      view or stratified exemplars (R23A, R24); the strongest photos hide what a unit does across its range (R23B).
    - The simulation-style scorer penalises the correct context reading (R33); the only scorer that ranked partly
      right readings and passed planted controls was predicted activation over a fixed pool (R19, L516).
    - The standing question set found facing because flips are an augmentation; content off its axes is invisible
      (R29 caveat; R30B).
3.4 Findings about the units.
    - 4e:55: cats and cars share the main input route 493 (29% of drive; L105-116); orientation-specific (R10, 4th
      percentile); pool tables fire (0.35), fireboats and flames do not (0.09-0.21), round Persian cats fire (0.48)
      (L484-487); no blind photo-level check backs "pointed drives, round suppresses" (R12); the essay's own
      bundle-plus-blind-maximum step files it with the planted mixtures (R17c, judge 1); V is the lowest reading of
      a real unit on every scorer (L515). The upright-face and downward-V readings are not separated by any test run.
    - 4d:409 is a real shared-pose unit (R13-14), and the essay's route-dependent render claim reproduces on it
      (R17b). Generalised pose units are rare under janus's coarse measure (R13), not "more than a hundred".
    - Zoom In's oriented dog heads reproduce at the 99th percentile across the range, 4 of 6 (R23B); his car-circuit
      "side view" labels show no within-class effect (R30A, L949); 5a:719's pinched shape is not visible (R20).
    - Channels beat random directions in every layer but directions are far from unreadable (R32); OLMo directions
      are often as readable as neurons (R33); a meaning across forms is decodable but not neuron-sparse (R34);
      early cores are usually unrelated to the final top contexts (R36).
3.5 What this says about "can a model read a unit his way". From the data alone, no, and the failure is not
    disposition: the reader lacks the comparison that makes a shared variable visible (peers, baselines, contrast
    pairs, the range), and when the comparison is measured and tabulated the reader's role shrinks to selection.
    The Fable consult said the same before the measurement rounds (OBSERVATIONS D4). What janus could not test is
    whether his wiring views, applied with judgment inside a labelled network, do what the data-side views did not;
    his own blind step, applied by janus, did not.

## 4. Recovering his labels as a calibration

4.1 What exists to calibrate against.
    - Qwen3-32B and the MoE model: 1000 labels with two scores and 7-10 blind test rows each, on open-weight models
      (OBSERVATIONS B). The notes are not public; the corpus is not public; which examples are his agent's own is
      not stated.
    - InceptionV1: the audited car-circuit labels (111 neurons, short strings), the V reading of 4e:55, the pinched
      shape of 5a:719, the pose claim, plus Distill's curve, frequency and dog-head IDs (OBSERVATIONS D5; literature
      IDs). janus has already tried seven of the car-circuit labels: four "side view" labels did not hold within
      class and three outside references were named by bare readings (R30A).
    - ConvNeXt medium and large: a browsable microscope and circuit atlas; whether the weights are released is not
      stated anywhere I read.
4.2 What "recovering" would mean. Three different endpoints, which the brief already separates and which should stay
    separate:
    (a) Agreement with his label text. Cheap and weak: a harness primed to find one thing agrees with him on
        everything, including the cases where he is wrong; janus's R17c shows his own step disagreeing with him.
    (b) Passing his tests. For the Qwen gallery the test rows are the label's own new examples and controls; a
        harness label can be scored the same way (new examples that fire, near misses that do not, distant texts
        that never do: L649-655). This measures the centre of the label, as he says, not the mixture question.
    (c) Instrument validity on units with known jobs: planted one-rule units with misleading surfaces (janus
        U-862, U-733), planted two-rule units (cheese/C++, bird/month, summed channels), noise, random directions,
        and shuffled weights. janus did (c) throughout; it is the part of calibration that does not need his labels.
    A calibration that only does (a) measures the prompt; (b) measures the centre; (c) measures the instrument.
4.3 How we would know a harness reads units his way rather than assuming they are clean. The evidence says the leak
    is not the assumption in the prompt: arms told the network is clean did not unify packed controls (R1, R6, R8).
    The leaks janus found are confirmation in what gets probed (R2, R9), the cheap "unrelated" exit (Fable consult),
    reading typical effects as distinctive (R11), and prose beyond the checked rows (R29, R30A). So the checks are
    behavioural: does the reading change when the view changes (crops, range, peers, readers) and not when the
    prompt changes; does a planted mixture come out as a mixture and a planted single rule with a misleading surface
    as a rule; does the harness place a unit against its family and report what is its own (R27 post hoc); does it
    ever disagree with his label and win on held-out activation (R19's scorer is the one that could tell).
4.4 Why this is harder than it looks. His tests are partly circular as a calibration (his agent wrote the examples
    that his label is scored on); the 1000 were "chosen randomly" but the dense-model half spans only 500 of ~1.6M
    poles; thresholds for "one job" are his; and his InceptionV1 labels, where janus could check within class, did
    not hold. A calibration that recovers his labels on Qwen and fails on InceptionV1 would tell us about the models
    as much as about the harness.

## 5. Where a better harness or kernel would matter

5.1 The cost centres, with what is measured so far (none by me).
    - Collection: forward-dominated and amortised; exactness and range coverage are the engineering problems
      (proteus: ties, non-i.i.d. survival, 60 min for 169k units). The field keeps top-k; his checks need the range
      (L333), the negative pole (L241), and co-activation for families (brief).
    - Reading loop: seconds and cents per call; the published costs are $0.046 per neuron all-in (Transluce) and
      ~$0.1-0.5 per janus session. Unknown for his papers.
    - Checks: scoring dominates the field's spend (5-30x the label); network-side checks (bundles, ablations,
      steering, held-out forwards) are a backward or forward pass each, per unit, and are absent from the field's
      pipelines, so their cost at scale is unmeasured.
5.2 Which would change what is possible rather than only the speed.
    - A collection pass that keeps enough per unit for the family, range and negative-pole views to be lookups is
      the change that made janus's decisive rounds possible at all (every unit-against-layer percentile needed a
      layer-wide cache; README L224-253, L868-911). That is a kernel and storage question, and it is where "exact"
      matters, because ties decide who is in the top set.
    - Batching held-out stimuli across units (one forward, many invokes) turns the scoring of hundreds of labels
      into one pass. That is speed, except where it makes a within-class check affordable that would otherwise be
      skipped, which is what happened in janus between rounds 15 and 29.
    - The within-unit loop: round 33 says the gain is in seeing the experiments, not designing them; round 30B says
      a fixed battery misses content. So the loop can be split: a standing battery in one pass for every unit, and a
      sequential loop only for the residue. That changes the shape of the workload more than its speed.
    - Readers and the crossword: nothing in any pipeline reads a unit through its readers, and his claim that
      understanding compounds makes labelling a frontier that sweeps a network. If that is right it is the one part
      that cannot be parallelised across layers and the part where a harness, not a kernel, would matter.
5.3 What is not known. The call count and cost of one of his neuron papers; the cost of attribution bundles at the
    scale of a 32B model; whether an exact atlas of a language model's MLP units at his corpus scale (10B tokens,
    L401) is feasible on one box; whether reading through readers helps at all (no experiment has tried it).

## 6. Questions I would put back

- Why did the essay's confirmations for 4e:55 not reproduce from weights (L140-152)? Is his figure attribution over
  natural images, as README L297 suspects? Without that, the one worked vision example has no public wiring test.
- Why does the brief put "the list hides it" where the essay puts "the nouns hide it"? Both are consistent with
  janus; only the second is his claim, and it is the one a harness could act on (ask for the variable that runs from
  the most negative to the most positive example, as the Fable consult put it).
- Why calibrate on his labels at all, rather than on tests his labels must pass? Round 30A suggests his InceptionV1
  labels can fail within class, and his Qwen tests are his agent's own examples.
- Why is scoring the field's largest cost when he says it is not the right place to start (L573)? If the checks
  that matter are the within-class and family ones, the cost model of a harness is different from the field's.
- Why has nobody read a unit through its readers? It is the step his definition of "clean" rests on (L477).

## 7. What I am not doing in this pass

No design, no cost model, no proposal. The brief asked for the problem first. The two places I think a design would
have to start are 5.2's split of the loop and the family/range collection, but that is for after these answers are
checked.
