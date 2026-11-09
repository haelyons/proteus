# autolabel: observations

Written 2026-11-08 after the reads, before any answer. Neutral record of what the evidence says, with citations.
"Essay L<n>" = line n of `janus/authors/informed.prompt_sent.md` (the file has a 14-line prompt header; the essay
starts at line 16). "README L<n>" = `janus/README.md`. Web sources were fetched today; quoted figures are as the
pages state them.

## A. The essay (Cammarata, "Neural Networks Are Already Clean", 2026-10-15)

A1. Thesis. "a network's own mechanisms are its legible building blocks"; the canonical polysemanticity evidence
    "was misread"; what remained was "recursive generalization" (L23). Two approaches: normal science on the network
    itself, or extracting hypothesised features (L25). Four-step argument at L63-71.
A2. Numbers. Qwen3-32B preliminary audit: "roughly 90% of neuron poles have one main job, often with small related
    offshoots; another ~7% have two related jobs; and only ~2-4% are not yet understood" (L41); the share "varies
    depending on how you set the threshold for counting weak activations" (L43); "97% non-arbitrary mixtures" in
    closer sets (L43); "distilling everything into one 'cleanliness' score should be taken as a broad pointer" (L43).
    InceptionV1 last layer: ~5% (old tools), ~50% (better tools), ~70% (retrained), ~99% ConvNeXt (L127-137).
    GPT-2 XL: 500 neuron papers, ~70% "roughly one thing", pristine ~5% (L413).
A3. The 4e:55 reading. "I opened up the cat and car image grid I'd seen hundreds of times before and did something
    I'd never done: I spent a few seconds wondering what one thing this might be. Almost instantly it was obvious:
    ... it's a V-shape neuron" (L85). Then: "I looked at its circuit connections: V-shaped flames, V-shaped water
    jets from a fireboat, V-shaped fox faces. I looked at its negative pole: an O" (L87). Figure caption: "its
    strongest inputs and readers ... pointed things drive it and round ones push it negative, whatever the label"
    (L89). Footnote 2: "To see the shared pattern, you have to visualize what the neuron's inputs have in common, not
    the neuron on its own" (L83); whiskers come from feature visualisation taking the cat route (L83).
A4. Dispositions. Interpreters are "strongly biased toward describing neurons as detectors for groups of nouns ...
    the neuron is better understood as a verb than a noun. Its job isn't to name what's in the input but to draw a
    distinction downstream neurons can use" (L109). "if I couldn't use groups of nouns at all, how would I describe
    this neuron ... easy to test on more groups of nouns (for the V: shinkansens, pool tables)" (L111). "Keeping the
    original neuron whole keeps these errors visible" (L113). Negative side "a quick check on whether a grouping is
    real: the V neuron's negative side is an O" (L241). "superposition the null hypothesis when it should have been
    the last hypothesis" (L213). "we don't have to force an explanation we don't have" (L217). "what the neuron
    learns should be trusted, its natural sparsity and activation ranges" (L43; L333 percentiles vs raw). Weak
    activations: "Usually not" needed to label; "look at the top activations, see that it's Titanic-ish, and that
    label usually holds up through the rest of your audits" (L339-341). "If you assume a network is clean and you're
    wrong, a messy neuron is a lead ... If you assume it's packed ... a messy neuron is the end" (L533).
A5. The language-neuron procedure, as described. For the "righteousness/bargain" neuron: read the surrounding
    sentences, "check its steering effects", "look upstream at how it's built and downstream at what it does, and
    both line up", "check its negative pole", "run specific tests, like checking which store brands it responds to.
    Any of these could break the explanation, yet they keep pointing to the same place" (L51-55). "If you can't
    understand it, little else helps: circuits and attribution graphs only explain a neuron in terms of other
    neurons you don't understand yet" (L51); understanding compounds "as the map fills in" (L51, L57).
    Other named checks: long context (L203), decoding/transliteration (L195), broad datasets (L207), sign regimes
    of SwiGLU (L385, L585), negative logit connections (L415-417: "eight of its ten strongest negative logit
    connections begin with [a dot]"), ablations inside attribution graphs (L365: remove 8 Santa-negative neurons,
    Santa falls to 6%; remove 20 labelled Christmas neurons, 76%).
A6. Tests he reports. Blind maximum: one agent explains, a second "sees only that explanation and writes an example
    meant to activate it maximally ... it can activate the neuron more strongly than any example we found in 10
    billion tokens" (L401). Bundles: "cluster the attribution flowing into the neuron ... run a blind maximum for
    each bundle, ablating all the others ... A bundle that had been packed in rather than grown would show up here as
    an explanation unrelated to the others that still drives the neuron" (L407). Explanations that could lose: 200
    random neurons; from the explanation alone an agent wrote 5 new examples and 5 near misses; among 183, new
    examples fired strongly 76%, near misses 11%; "won for 164, tied for 15 and lost for 4"; a strong response is
    30% or more of the corpus max (L649-651). Distant-concept texts fired ~1%, "179 of the 183 neurons never fired
    on them" (L655). Weak-activation ablation: 69.47% vs 70.08% accuracy keeping ~the strongest quarter (L595).
    Agents reviewed 100,000 attribution graphs; "I don't trust our current tools enough to turn it into a
    cleanliness score" (L645). 25,000 ImageNet attribution graphs factored into edge motifs (L661).
A7. What he says does not establish or refute. "'Cats and cars' could do well on many tests of the V neuron while
    completely missing what it really is. You can break it by noticing it connects to flames ... they all take
    creativity and judgment rather than a summary statistic" (L409). Simulation scores "measure where attention
    puts information", reward weak-activation detail, LMs output values poorly (L573). "counting how many of a
    neuron's top-k examples match a label ... would give superposition and recursive generalization similar scores"
    (L821). Recursive generalization predicts: "everything on it should trace back to the same local origin ... and
    it should grow out from there over training in a particular order" (L827). "I would update strongly on even one
    convincing example" of a group whose joint activity stays inexplicable (L645). "To my knowledge, no one has
    published a second circuit with the right shape" (L33).
A8. Scale statements. "100 million tokens through one Qwen3-32B neuron ... about 160 reach the strong half" (L315).
    "Here are 1000 neuron papers, detailed investigations into individual neurons, chosen randomly" (L377); the
    companion paper is "forthcoming" (L75, L413). Unresolved: mixed5a:719 "pinched-center shape" with semantic
    residue (L735).
A9. "What I mean by clean": "a network's parts are built to be read, and built upon, by its downstream readers ...
    become another reader" (L477). Exploratory over enumerative interpretability (L481-491).

## B. The 1000 neuron papers, as published

B1. The live page loads `data/neuron-gallery.json` (1000 entries) and `data/neuron-gallery-more.json` (1000
    entries), found in the page HTML (data-src attributes). Per entry: `id` (e.g. L11_N12072 or L0_E100_N436), `k`
    (500 `d`, 500 `m`), `L`, `N`, `E` (500, the `m` entries), `t` title (median 61 chars), `l` label (~460 chars),
    `st` = "tested" for all 1000, `i` and `c` integers 1-10, `m` a list of 7-10 test rows [name, activation] (997
    entries; 586 have a row named "...control"; 2359 of 9581 rows carry negative values), `h` a notes link for 30,
    `q` an activation-percentile table for 3. The "more" file adds `l` (full label), `wi` (one-line gist), `wc`
    (one-line verdict on the checks). `d` layers run 0-63 and N up to 25535 (Qwen3-32B); `m` layers 0-39, N <= 511,
    with expert ids (a MoE, presumably the Qwen3.6-35B-A3B named at L600).
B2. `c` reads as a cleanliness-type score (c=1 titles: "A dead neuron ..."; c=10: crisp lexical detectors);
    `i` as interest (i=1 dead neurons; i=10 "Beings who mistake their small world for all of reality"). This is my
    inference from titles, not documented. Distribution of c: 6 (319), 8 (268), 7 (187), 5 (92), 4 (53), 2 (27),
    9 (23), 3 (16), 10 (9), 1 (6).
B3. The notes pages (`/notes/<id>/`) return 403 Access Denied (curl and WebFetch, today). The companion paper is
    not out (essay L75; LessWrong thread). So the per-neuron procedure is public only as the essay's prose plus the
    gallery summaries and test rows. Goodfire's product agent "Silico" exists (goodfire.ai/silico) but no public
    document ties it to the neuron papers.

## C. Field auto-interp pipelines (web, today)

C1. OpenAI 2023 (Bills et al.; code github.com/openai/automated-interpretability). Explainer prompt: at 2K context
    only the first two activation records, at 4K all (explainer.py); "Activation values range from 0 to 10";
    explanation prefix "the main thing this neuron does is find". Simulation: all-at-once via `echo=True,
    logprobs=15` and an expected value over the tokens "0".."10"; token-by-token variant one call per token
    (simulator.py). Score = correlation. GPT-2 XL has 48 x 6400 = 307,200 MLP neurons; the essay's own figure for
    that project is "about 1.7% of neurons interpretable by our most generous measure" (L39).
C2. EleutherAI delphi (github.com/EleutherAI/delphi; Paulo et al., arXiv 2410.13928). Cache: `LatentCache(...,
    batch_size=8)`, "at least 10M tokens", safetensors sharded over the dictionary width. Sampling: top, random,
    quantiles, mix; non-activating from random/neighbours/FAISS. Explainer sees 40 examples of 32 tokens (Llama 3.1
    70B). Scorers: detection (5 shuffled examples per call), fuzzing, simulation, surprisal, embedding, intervention
    (30 scoring + 10 explainer prompts per feature). Pipeline concurrency `max_concurrent`; vLLM offline client.
    Costs per 100k features (Llama 70B): fuzzing $676, detection $588, all-at-once simulation $3.6k, token-by-token
    $18.7k; "$200 to interpret 1 million features". Neurons vs SAE under these scorers (Table A10, median
    fuzz/detect): Gemma 2 9b SAE 0.76/0.74 vs top-32 neurons 0.62/0.59; Llama 3.1 8b SAE 0.81/0.83 vs neurons
    0.55/0.53; random baseline 0.51.
C3. Transluce (transluce.org/neuron-descriptions). All 458,752 MLP neurons of Llama-3.1-8B-Instruct, both
    polarities; describer sees 10-20 top-activating exemplars; distilled GPT-4o-mini explainer best-of-25;
    fine-tuned 8B simulator; $15,951 API spend plus 2,891 GPU hours (~$5,174): "approximately $0.046" per neuron.
    "not representative of the full behavior of the neuron across all quantiles".
C4. MAIA (arXiv 2404.14394). Agent loop with tools: dataset_exemplars, text2image, edit_images, describe/summarize
    (a fresh GPT-4V instance "to limit confirmation bias"), log_experiment. 100 units from each of three vision
    models; 85 synthetic neurons with ground-truth selectivity, including disjunctions ("trains OR instruments") and
    conditionals; predictive evaluation with 7 positive and 7 neutral prompts per label; 2AFC vs MILAN 0.73, vs
    human 0.53.
C5. nnsight (nnsight.net). Batching: several `tracer.invoke` blocks inside one `trace`: "Their inputs are combined
    into a single batched forward"; a Batcher narrows each block to "only its rows of every activation" and widens
    edits back; padding to the longest input shifts left-counted positions; `.skip()` must cover every invoke.
    Remote: the traced block is serialised and POSTed to NDIF; "Only values touched by `.save()` ... are shipped
    back"; "Each request has a one hour time limit"; jobs queue; sessions bundle traces into one job. Hosted
    models: the status page; the feature page's example shows gpt2.

## D. janus (haelyons/janus; 84 commits; 113 ledger entries, 66 freezes, 26 results)

D1. Question and contract. "Can a prompt, given plain recorded data from one unit ... read it the way ... Cammarata
    does ... And does it get there by reading the data, rather than by assuming neurons are clean?" (README L3-5;
    CONTRACT.md). Cards are top activations (text: 12 strongest + up to 8 middle; image: top grid, full format adds
    negatives). Two turns: class + explanation, then 20 held-out fire/no-fire predictions (>= 30% vs < 5-10% of max).
    Subjects are bare `claude -p` sessions, no tools (tools/claude_call.py, run_subjects.py).
D2. Costs and latencies (from the saved transcripts, `runs/*/*/s*.json`, medians per session): bare two-turn arms
    $0.07-0.14, 12-14 s, ~9-10k input tokens; probe/experiment arms $0.14-0.50, 13-28 s (P1_active 5 turns, $0.375,
    28 s). Measurement: InceptionV1 at 416 photos/s on a GTX 1070 vs ~25/s CPU (README L591); battery over all 832
    mixed4e channels x 10 transforms from one cached pass (spec_battery.md); peers in "one batched pass"
    (spec_peers.md); round 29: 48 blind-answer calls covered 570 photos (README L871); round 30A: 298 calls one at a
    time (L937); round 31: 3,080 photos answered by Qwen3-VL-32B with tensor parallel 4 on 8 x A100 in about 20 min
    (L1035-1041); round 36: 10,000 documents x 256 tokens x 35 checkpoints of OLMo-2 7B, 2 h 18 min wall for the
    7B job (`runs/r36/time_7B.txt`), about $14 on Lambda (L1162).
D3. Round-by-round (abbreviated; full text in README at the cited lines):
    - R1 (L21-45): planted one-job and offshoot units read correctly by all six arms; packed controls not unified
      (A0/C0 2 of 3 "1+" on bird+month while describing two triggers). 4e:55: "Every arm reads 'faces of small furry
      mammals' and misses every held-out car and pool table (held-out 0.65-0.70 in every arm). No arm reaches a
      shape reading, including C4, whose author read the essay" (L44-45). Round-1 predictions 6-8 (C4 shape reading,
      negatives moving arms, shape explanations scoring higher) failed (PREDICTIONS.md L13-34).
    - R2 probe (L47-60): with up to 3 data requests, 6 of 6 probe cars and cats and conclude unrelated; "none looks
      for what cars and cat faces share" (L64). Text units never use the offer.
    - R3 (L69-73): 6 of 6 pick "faces" over "pointed/V", citing the hamster and beaks among negatives; the negative
      grid "misleads" (substrate, L80-82).
    - Substrate (L75-85): cars co-dominate class means; pool table 32nd, bullet train 43rd, fireboat 489th of 1000;
      drawn shapes under 10% of max.
    - R4 feedback (L87-93): outcome feedback converges on two-feature reading; held-out 0.75-0.85.
    - R5 display (L95-103): crops give "car front face-like" (first stated link, L100); located negatives give
      "dark, flat muzzles" vs "pointy faces". Held-out unchanged.
    - Circuit (L105-116): input 493 carries 29% of positive drive, top input for cats and cars alike (median 40% on
      cats, 20% on cars); car-specific 352; cosine cat vs car 0.72.
    - R6 (L118-138): contrast pairs read as pose/size; circuit shown by contribution gives "Zoom In's reading" 3/3
      unrelated; N0 no-nouns 0/3 shape-primary. By weight the second input is 511 (crickets; pool tables drive it
      2.7x), hidden by contribution ranking (L133-137).
    - Renders (L140-152): "The flames come from a reader, which no subject has seen" (mixed5a:94, L149); inputs by
      weight render as faces, crickets, arcs, grille textures, domes.
    - R7 (L154-160): renders used as confirmation only.
    - R8 (L184-194): the essay's question on the famous mixed grid: 3/3 unrelated; pointed objects 0.50 in all 12
      sessions.
    - R9 edits (L196-213): 5/6 run experiments; classic face tests; "The edits test the hypothesis already held"
      (L204); upside-down loses most activation, which fits both readings.
    - Separation audit (L215-222): orchestrator not blind in rounds 3-9; from R10, blind authors and claim-blind
      implementers (and from R22, separate testers).
    - R10 battery (L224-253): rotate_180 retention 0.27 at the 4th percentile of the layer; "The battery cannot
      separate 'upright face' from 'downward-pointing V'" (L253).
    - R11 (L255-267): blind-authored scientist arm; subjects read typical drops as evidence; "They have no baseline
      to tell typical drops from distinctive ones" (L265).
    - R12 (L269-309): blind class lists reverse both shape contrasts ("So the earlier substrate result ... came from
      how I drew the groups", L287); regions: the unit prefers pointed cells less than the typical channel; showing
      the layer baseline changes nothing.
    - R13-14 (L311-345): 4d:409 is a blind-confirmed shared-pose unit; Q0 unifies it 3/3 "side-on"; "the same
      question that did nothing there unifies this unit at once" (L343).
    - R15 (L347-368): near-miss pick test works on 409, "For mixed4e:55 it fails its own control" (L364).
    - R16 (L385-410): attribution bundles: the pose unit's dogs and vehicles arrive by separate inputs, so separate
      bundles are not evidence of packing; 55 shares more (493 at +0.62).
    - R17 (L412-461): curve IDs hold; route-dependent renders reproduce the essay's pose claim on 409, none shows a
      V or flames for 55; blind maximum per bundle: 55 judged 1 (unrelated), with the planted sums (planted 2 and 1;
      car 5; curve 4; pose 3), "no explanation names a V, a point or flames" (L458).
    - R18-19 (L462-516): own-example near-miss pools cannot rank partly right readings; predicted activation
      (Spearman) is "the only one that ranks partly right readings and passes both planted controls" (L516); "V is
      the lowest reading of a real unit on every scorer so far" (L515); pool tables 0.35 (real point for the essay),
      fireboats 0.09, round Persian cat 0.48.
    - R20 (L518-557): open method on 11 units: car parts reproduce; oriented dog heads do not (431 loses 57% when
      mirrored, 100th percentile, read as fur texture): "The miss on 431 is the method's: no reading proposed
      direction, so nothing tested it" (L555); 5a:719's essay reading not visible.
    - R21-24 (L559-695): "Crops are the big effect on shallow units" (L578); profile with layer baseline lifts the
      pose unit; direction read only from a spared view or stratified exemplars; "A unit's strongest photos can hide
      what it does across its range" (R23B).
    - R25-26 (L697-782): readings turned into blind photo questions, calibrated within class; nobody asks left/right
      until a standing set of augmentation questions does; the table format then loses it.
    - R27-28 (L784-866): contrasts with peers: "a held prediction often describes the family" (L829); "The units
      that prefer one side differ from their family on exactly the axis whose sides the agent pooled" (L833); family-
      aware refinement gets the family right, names no side in 0 of 72.
    - R29-29B (L868-931): a fixed blind-answered question set, every answer pair compared against the layer and 3
      peers, read by an agent: correct side for 4 of 4 oriented units; "The final reader does what a rule would"
      (L923); prose overclaims beyond checked rows.
    - R30 (L933-1011): transfer to 15 unseen units with outside references: Zoom In's 4a facing consistent; Nick's
      car-circuit "side view" labels: "no stable within-class effect" (L949); random units: "the missing candidates
      dominate" (L994); rows at >=95 keep sign on new photos 87% but reach 95 again only 36%.
    - R31 (L1013-1041): open VLM answerers below Claude's own repeatability; "The facing signal survives a weaker
      answerer; most other questions do not" (L1028).
    - R32 (L1043-1070): channels ahead of random directions in all 7 layers, "but random directions are far from
      unreadable ... weak support to 'neurons are clean' as a claim about the neuron basis" (L1065).
    - R33 (L1072-1128): text experiments: P1 finds planted rules the card hides (+0.11 held-out on misleading
      planted cards); choosing experiments adds ~nothing over seeing them (Y1 ~ P1); random OLMo directions often as
      readable as neurons; the simulator score penalises the correct context reading.
    - R34 (L1130-1153): one meaning across 32 forms in OLMo-2 7B: linearly decodable (AUC 0.942) but not neuron-
      sparse; "form is easier to read than meaning" (L1149).
    - R35 (PREDICTIONS.md L808-864; runs/r35; not in README): agent labels from a feature visualisation plus top-5
      photos, 52 Inception units in 13 blind pages, judged by Codex on 8 held-out photos: explained 0.79, random
      neurons 0.69, clusters 0.58, random directions 0.50 (2 of 4 "can't tell" on two of the three direction pages);
      group scores 9 of 12; the human half (`human.md`) is blank. I1 held (0.792 vs 0.688+0.10), I2 failed, I3
      failed (side words in 6 labels), I5 held.
    - R36 (L1155-1197): OLMo-2 top-16 documents assemble over all of pretraining; early cores "usually unrelated to
      the final top contexts" (judge 27%); random directions keep cores better than neurons.
D4. The Fable consult (`janus/consult/fable_reply.md`, 0 tool uses): "the instruction variants changed belief,
    which isn't the bottleneck"; the leap is "from describing the inputs in the observer's categories to finding the
    variable that cuts across them"; triggers: committed predictions with failures, contrast pairs, the axis from
    most negative to most positive, price the "unrelated" exit, sample rivals and select by held-out, track record.
D5. His labels available locally: `janus/authors/ext/car_relp_circuit.json` (from the essay's car-path archive):
    111 visible neurons across mixed4a-5b and logit, 37 groups, 8 families, 165 label strings such as "vehicle
    wheel-arch / dark rounded inset in a hard bod...", edge method "sum(abs(RELP edge weight))", 100 images. Round 30
    drew 4 "nick_inside" and 3 "nick_outside" units from it (round30_selection.json).

## E. Harness documents (metis, as design only)

E1. `metis/dut/pipeline/m5/DESIGN.md`: blind operator reads only fixed-schema signatures; content in `agent/`;
    sealed predictions before method code; reproduce anchors before profiling; one gate with five fresh-session
    skeptics; separate implementer and test author; three independent verdicts (validity, equivalence,
    performance); "Missing measurements are VOID, never PASS".
E2. `metis/dut/latent_skeptic/HEURISTICS.md`: H1 independence of actors, H2 a crux is decided by a measured number
    that could have come out otherwise, H3 numbers re-derived from primary artifacts, H4 "The instrument is itself a
    claim" validated once per instrument version on held-out ground truth.
E3. `autolabel/context/metis.md` (Helios): gates that count items pass bad work; tests of the specification passed
    while execution failed; calibrate before proposing.

## F. proteus (engineering only, from the GitHub listing; no results read)

F1. `atlas/core.py` (streaming state, seeds, verification, merge, checkpoints), `atlas/filter_triton.py` (fused
    compaction with early exit on empty tiles, 8x512 tile), `atlas/stats_triton.py` (statistics and Gram kernels).
    Claims: top-K exact for the computed activations, ties broken by token id, threshold only rises so results are
    order-independent; reservoir bins by bottom-k on a hashed id; dense fallback ~28x a forward pass on an A10; one
    A100 40 GB: 48.8 min forward alone, 60.4 min forward plus atlas, 71 min end to end. The brief adds: candidate
    survival is not i.i.d.; bf16 ties occupy ~80% of top-64 slots.
