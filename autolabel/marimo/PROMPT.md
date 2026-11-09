# Brief: a marimo notebook for a first read of one unit

From Claude (the autolabel session), for Codex, 2026-11-08. The owner wants a first sketch of the marimo notebook that
lets us do a first read of a unit: the views a reader looks at, and the reading it records, before any automated
pipeline is built. I take "first read" to mean that; say in REQUIREMENTS.md how you interpret it.

Working directory: /home/hal/dev/interp.

## Read first
- `autolabel/BRIEF.md` (the owner's context and boundaries), `autolabel/ANSWERS.md` (the current design thinking,
  especially sections 0, 2.3, 2.4, 3.5 and 5), `autolabel/OBSERVATIONS.md` (evidence with citations),
  `autolabel/PREDICTIONS.md`.
- The essay: `janus/authors/informed.prompt_sent.md` (Cammarata, "Neural Networks Are Already Clean").
- `janus/README.md` and `janus/CONTRACT.md`. The rounds that matter most for views: 10 (battery with layer
  percentiles), 21-24 (crops, invariance profile, spared view, stratified exemplars), 27-29 (peers and the
  every-pair table), 32 (channels against random directions).
- A recent Codex answer comparing our approach with an external neuron-labelling paper:
  `~/Documents/Remote/interp/xtal/astra_ANSWER.md`.
- Infrastructure to reuse rather than rewrite: `janus/tools/battery.py`, `peers.py`, `rangeprofile.py`,
  `pairtable.py`, `circuit.py`, `featvis.py`, `bank/img_acts.py`, `bank/olmo_units.py`; caches under
  `janus/runs/battery`, `janus/runs/proploop`, `janus/bank/img`, `janus/runs/r33/olmo_acts`.

## Deliverables, in order (all under `autolabel/marimo/`)
1. `REQUIREMENTS.md`: what a first read must show and record, and why. Trace each requirement to the evidence
   (essay line, README round, file:line). For each, say which existing cache or tool provides it, or that nothing
   does yet. Separate what a unit responds to from how it is wired, as ANSWERS.md does.
2. `DESIGN.md`: a short sketch. Data layer (what loads from caches, what is computed live and at what cost), the
   notebook's cells and views, how the reader's reading is recorded (for example a prediction committed before a
   reveal), and what is deliberately left out of a first version.
3. `first_read.py`: the marimo notebook implementing the first version. Start with InceptionV1, where the caches
   exist; add the OLMo text units only if it is cheap. It must open with
   `janus/.venv-gpu/bin/marimo edit autolabel/marimo/first_read.py` and also run top to bottom as a script without
   error. Check it on at least two units with different status: `mixed4b:431` (known: right-facing dog heads) and
   `mixed4e:55` (contested). Report what you ran and what came out in a short `NOTES.md`.

## Boundaries
- Write only under `autolabel/marimo/`. Read janus and its caches; never modify them. Do not open `metis/experiments`
  or `metis/ops`.
- No Lambda or other paid GPU launches, no spend, no commits or pushes. The local GTX 1070 (8 GB) is fine for small
  forward passes. Never print or copy anything from `dm_hack/.keys`.
- Write the three documents in plain English for the owner, with citations. Keep the design small: subtract before
  adding.
