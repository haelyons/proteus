# autolabel: brief

From Helios, 8 Nov 2026. This is the context I'd otherwise give you in person before we start.

## Where I am

I'm converging on a goal: parallelise automated unit labelling. nnsight and the tooling around it already carry heuristics and approaches for this. Recovering Goodfire's 1000 neuron investigations (Cammarata, "Neural Networks Are Already Clean", 15 Oct 2026) makes it clear there is room for better harnesses and better kernels. We've covered that part reasonably well. What we've only partly covered is what it means to apply Cammarata's approach, and I'm lost in the noise of the documents and chats around it. This session is for that.

## The essay

- Full text: `janus/authors/informed.prompt_sent.md` (the copy sent to a janus author; it includes the section introducing the 1000 neuron investigations, and the references). Site: https://already-clean.goodfire.com/
- Our reading so far: the famous polysemantic units were misread, because a list of top examples hides the one feature that links them. If that holds, the single unit is a usable handle on a model's concepts. "Clean" in his sense is meant as deeper than neuron monosemanticity: a network's own mechanisms are its legible building blocks. Check this reading against the text; don't inherit it.

## Prior attempts, in reading order

1. **janus** (`~/dev/interp/janus`, GitHub haelyons/janus, 20–30 Oct, no biology). Can a prompt, given plain recorded data from one unit, read it the way the essay does, and get there by reading the data rather than by assuming cleanness? 36 rounds. Rounds 1–32: InceptionV1 mixed4e:55 over ImageNetV2, other InceptionV1 units, and planted text units over Pile-10k. Rounds 33–36: OLMo-2 text units, one meaning across many forms, and neuron growth across training checkpoints. `README.md` has every round's design and result; `PREDICTIONS.md` was sealed before each round and `LEDGER.jsonl` holds the hashes; `CONTRACT.md` defines cards, arms, truth, scoring and isolation; `tools/` has one script per round. Early signal: every arm read the cat-face-plus-car unit as "faces of small furry mammals" and missed held-out cars and pool tables; given the option to probe, agents rediscovered the field's 2017 reading (two unrelated jobs) rather than a shared shape. My notes from that period: `context/janus.md`.
2. **metis** (`~/dev/interp/metis`, private, 17 Oct–5 Nov). An agent pipeline that turns research questions into GPU experiments with a blind operator, sealed predictions, split judges (Codex and Claude) and gates. Relevant here only as harness design: `dut/pipeline/CONTRACT.md`, `PILOT_PLAN.md`, `PROMPTS.md`, `m5/DESIGN.md`, `m5/SCHEMAS.md`, and `dut/latent_skeptic/HEURISTICS.md`. Lessons from the operator threads: gates that count items rather than content pass bad work; an operator holding most roles approves its own departures; blindness buys independence but costs scientific context; calibrate before proposing. My notes: `context/metis.md`.
3. **proteus** (GitHub haelyons/proteus, public, 3–7 Nov). Methodologically the closest to what I want, but its subject is biological, so use it as engineering only. It built an exact atlas of every FFN unit's top activations in a 650M-parameter transformer: 168,960 units over about 194M input positions in 60.4 minutes on one A100, bit-exact, about 12× faster than baukit's top-K, with a streaming top-K and a Triton compaction kernel ("filter, don't select"). Engineering and method lessons: candidate survival is not i.i.d.; bf16 top-k sets are ill-defined because ties dominate (80% of top-64 slots); a ruler of descriptive categories measured correlates, not causes; a shuffled-weights control is essential; a trivial local baseline can match what looks like learned structure. If you need the code, read only the atlas and kernel source (for example `atlas/core.py`, `streaming_topk.py`, `filter_kernel_triton.py`; check the repo for exact paths), not `results/` or the README.

## What my own image-model reproduction taught me (my words, generalised)

- Our earlier ruler measured what a unit responds to; his method is about how a unit is wired. We had built half the test.
- Comparing a unit with its closest peers worked better than anything else. Many findings turned out to describe the whole family, and each unit differed from its family on exactly one axis. A scorer that rates fifty near-identical detectors as fifty clean units is measuring the wrong thing.
- Build families by co-activation (the Gram matrix) and separately by input and output weights, and note where the rankings disagree: ranking by weight and ranking by contribution on data gave different answers.
- Look downstream: what a unit's output promotes or suppresses, and which later units read it. In my runs the decisive clue was in a reader, not an input.
- Test the instrument on units with known jobs first (planted one-job units, deliberate two-job mixtures, noise). If it can't call those, nothing it says about real units means anything.
- Write the decision rule before any number: what pattern of results makes us say units are clean, latents are, both, or neither.
- Separate modes in a unit's examples don't prove separate meanings; the family comparison is the check. When his own blind step was applied to his own famous unit, it filed it with the planted mixtures. The reading has to be blind, and predictions committed first.

## The parallelism angle

- The SoTA Scientific Parallelism Hackathon (21–22 Nov) rewards reshaping a workload, a credible baseline and correctness; a well-explained negative can still score (`context/harry.md`). That's the frame: which parts of labelling are sequential, and must they be?
- Candidate cost centres, to be measured rather than assumed: activation collection (exact top-k over a corpus), the reading loop (many cheap model calls per unit), and the checks (held-out prediction, near misses, families, downstream readers).
- Tools in play: nnsight (instrumentation and batched invocations; NDIF for remote models), vLLM for open readers (janus round 31 ran Qwen3-VL-32B with tensor parallelism), Lambda GPUs.

## How I work

- Register predictions before looking; commit them; score against them.
- Observe neutrally before updating. Cite files with line numbers, commits and URLs. Say what is uncertain.
- Claim-blind authors and blind readers wherever judgement could leak.
- As simple as possible, but no simpler. No solutions before the problem is understood.
- Push back on me when the evidence says I'm wrong.

## Boundaries

- **Biology stays out of this pass.** Fable 5.1's safeguards flag biology-heavy context and stop the reply, which is what derailed my last attempt. Don't open `metis/experiments/`, `metis/ops/`, proteus results or README, or material about specific biological systems. If a question needs biology, stop and tell me; we'll apply the principles there later, deliberately.
- A hook keeps every session out of metis's blind experiments, and it also denies Grep or Glob over any directory above `metis/experiments` (including `~/dev/interp` itself). Search `janus/`, `autolabel/` or a specific path instead.
- No GPU launches, spend, commits or pushes without asking. Credentials live in `dm_hack/.keys`; never print or copy them.
- Write only under `autolabel/`.
