# 650M layer 15 unit 2087: when the input carries repeat information, does changing this signal alter how the model uses it? (DRAFT, nothing frozen)

**Status, 2026-11-06:**
- Only the timing and memory pilot has run (below).
- The discovery pilot, which shows whether the behavioural contrasts have room to move, comes next. After it,
  the constructs, readouts and effect sizes are frozen in a later section of this file, in a commit made before
  any fresh base sequence is scored.
- Scope: construct forwards and existing atlas metadata only.

## What the existing evidence supports

1. **What makes the unit respond.** Suggestive, confounded.
   - Ubiquitin x1-x9 raises H68 of copy 1 from -1.34 to 6.68. Five held-out 80-residue fragments rise
     monotonically from x1 to x2 to x4.
   - But length grows with copy number, and the fragment test used the maximum over the construct, which has more
     positions to rise at.
   - Against length alone: two single-copy fusions (UBA52, RPS27A; 128 and 156 residues) stay at -1.4. That is
     only two sequences.
   - The values are known at H68 (ubiquitin) and at each construct's maximum. **How many positions of a repeat
     respond, and whether the unit responds at a masked position, is not known.** This matters for where to read
     out.
2. **What its output changes.** Unknown. The one test (Ub x4 silencing, 11 masked positions, one base) was null,
   with no record of baseline confidence or of the unit's activity after masking. Neither a ceiling nor
   redundancy was shown.
3. **Whether the change helps prediction.** Untested.
4. **Why the case is still well bounded:**
   - repetition is an input property we set exactly;
   - the hidden residue's answer is known, and the visible copy can be made to agree or disagree with it;
   - the unit's response is large (about q50 to beyond q99.999) and generalised to sequences it was not selected
     on;
   - a masked LM should copy strongly from an intact repeat, so the behaviour has room to move. The pilot checks
     this before anything is frozen.
5. **Caveats:**
   - Copying may be carried by attention with no need for this unit; the unit may sit downstream of that match.
   - The unit may respond at only a few residues of each copy (H and P dominate its peak). The intervention
     readout is therefore reported by the unit's activity at the target and its counterparts, not only averaged.
   - The 40 ubiquitin-dominated columns were selected through the same clue that first misled us about 2087. They
     are candidates, not a family.

## Already inspected (everything below was seen before this plan)

- **Atlas #2, unit 2087:** top-64, reservoir bins and edges (`ubq650_unit_predictions.md`).
- **The 40 exported columns** with ubiquitin share >= 0.48 of their top-64: layer, pole, share, top-64 values,
  edges (`infra/runs/atlas-ru-20261105-1704/ubq650_units.json`). Layer 15 has 2087, 3486 and 4165.
- **All probes in `ubq650_unit_20261105/`:**
  - ubiquitin, NEDD8, 61-75 transplants into 10 proteins, 1-50 shuffles, point mutants, Parkin UBL, SUMO1 / RAD23A /
    ISG15, 20 composition shuffles;
  - UBC (9 copies), UBB, UBA52, RPS27A, synthetic Ub x2 and Ub + 40 + Ub, SUMO1 x2, lysozyme 19-98 x2, peak
    samples' repeats;
  - ubiquitin x1 / 2 / 3 / 4 / 6 / 9;
  - held-out fragments P90745, G2X1Q3, Q4WN10, Q9I485 and I1RLL3 (x1 / 2 / 4), **excluded from the fresh set**;
  - the Ub x4 / x1 silencing run.
- **Timing pilot (smoke, 2026-11-06):** one non-held-out 80-residue block (seed 0) repeated 2 / 4 / 8 times. Logits
  were computed and discarded, never printed or saved. `ubq650_use_20261106/pilot_timing.json`.

## Draft design (to be frozen after the discovery pilot)

**Bases.** 80-residue windows with no internal 6-mer repeat.
- *discovery:* 8 non-held-out windows, seed 2.
- *fresh:* 48 held-out windows, seed 1, excluding the five fragments above.
- Ubiquitin is reported separately as an inspected named base.

**Constructs.**
- Fixed block length 80.
- The reference block R is never altered. Every other block is a copy of R, or a composition-matched scramble of
  R (seeded permutation, 2 draws), occupying the same positions.
- Behaviour, primary: B = 2 blocks, in both orders [R|X] and [X|R].
- Screen and patching: B = 4 blocks, R in slot 1 (and slot 4 for the screen), c = 0 / 1 / 2 / 3 copies with the
  remaining slots scrambles. Length stays constant while repetition varies.
- Deletion (R alone) is a secondary comparison.

**Targets and masks.**
- 8 positions per base in R, residues 4-75, seeded; one target masked per forward.
- *Agree:* the copy is visible and shows the true residue t at the corresponding position.
- *Disagree:* the copy is visible with residue v at the corresponding position. v is assigned so that each of the
  20 residues appears as v about equally often, and v != t.
- *Masked-all:* the target and its counterpart in every other block are masked; the scramble comparison masks
  the same absolute positions.

**Readouts** (stored raw, per forward):
- log p over the 20 residues at the target;
- baseline confidence: p(t), max p and entropy;
- 2087's pre-GELU value at every position, including the masked target and its counterparts;
- for screen units, values at reference-block positions.

Derived:
- copy advantage = log p(t | copy) - mean over scramble draws of log p(t | scramble);
- copying preference = [log p(v) - log p(t)] with the disagreeing copy, minus the same with the scramble. This is
  fixed alongside log p(t), because following the copy can lower p(t) and still be use of the copy;
- masked-all advantage;
- deletion contrast.

**Uncertainty.**
- Per-base means; 95% bootstrap intervals over bases (2,000 draws); a sign test over bases.
- Residues and variants within a base are never treated as independent.

**Screen.**
- Units, collected in the same forwards:
  - the 40 exported columns;
  - a comparison sample of 4 random positive-pole columns per layer (seed 20261106; 132 columns);
  - all 5,120 layer-15 units in reduced form (reference-block mean, target and counterpart positions).
- Response = values at reference positions for c = 3 minus c = 0 at constant length, plus the c = 0..3 slope,
  unmasked and under the prediction masks. Normalised by each unit's own spread across scramble constructs, so
  no atlas pass is needed.
- The layer-15 candidate group for patching is chosen on discovery bases by a rule frozen before the fresh run.

**Intervention: patching in the repeat construct.**
- The recipient is the copy construct; the donor is the matched scramble construct with identical masks.
- Patch point: layer-15 post-GELU. Positions: primary, every residue token of the construct (CLS / EOS untouched);
  secondary, R only and copy blocks only.
- Graded replacement toward the donor value: alpha = 0.25 / 0.5 / 0.75 / 1.
- Then the layer-15 candidate group (alpha 0.5 / 1). One layer at a time.
- **Controls, matched in number, placement and the magnitude of the combined residual change:**
  - 4 single layer-15 units patched from the same donor, whose change norm ||delta g x w_out|| over the patched
    positions is closest to 2087's in that item;
  - 2 random directions added to the FFN output at the same positions, with the same per-position norm;
  - for the group, 4 random same-size layer-15 groups whose *combined* change norm (summed vectors, so
    cancellation counts) is within 10% of the group's.
- Recipients: B = 2 agree; B = 2 disagree; B = 4 with c = 3.
- The reverse patch (repeat values into a single-copy input) is optional and, if run, reported as ambiguous
  either way.
- Readers and connections are left until a behavioural contrast is worth explaining.

**Conclusions kept separate:**
- *moves predictions:* distribution change at the target beyond controls;
- *helps prediction:* log p(t) falls in agree when the repeat state is removed, beyond controls;
- *reduces copying:* the preference shift in disagree shrinks;
- *accounts for part of the repeat effect:* the fraction of the copy advantage removed, beyond controls.

**Decision structure, with provisional numbers** (the pilot sets them, frozen before fresh):
- **Behaviour:**
  - *effect:* the bootstrap interval of the per-base mean lies above delta_b;
  - *excluded:* the interval lies within +-delta_b;
  - *inconclusive:* otherwise.
  - Provisional delta_b = 0.1 nats.
- **Intervention:** f = fraction of the copy advantage removed by the alpha = 1 patch.
  - *effect:* the lower bound of f exceeds delta_i and the matched controls' upper bound;
  - *excluded:* the upper bound of f is below delta_i;
  - *inconclusive:* otherwise.
  - Provisional delta_i = 5%. Raised if the pilot's control spread is larger.
- **Null wording:** if behaviour does not change, that closes this route to testing use; it does not settle
  every possible role for the unit.

## Timing and memory pilot (2026-11-06, this container: 4 CPU threads, fp32, EsmForMaskedLM)

| construct | tokens | s per forward (plain / with capture hooks / with patch hook) | peak RSS |
|---|---|---|---|
| 2 blocks | 162 | 0.58 / 0.56 / 0.51 (batch 1); 0.50-0.53 per sequence (batch 4) | 3.5 GB |
| 4 blocks | 322 | 0.92 / 0.99 / 1.01 (batch 1); 0.97-1.03 (batch 4) | 3.6 GB |
| 8 blocks | 642 | 2.24 / 2.09 / 2.23 | 3.6 GB |

The hooks add no measurable time. Batching does not help on CPU. Throughput is about 300-330 tokens/s.

## Estimated cost (at 0.52 / 0.97 s per 2- / 4-block forward)

| part | discovery pilot | fresh run | tokens (fresh) | A10 (extrapolated) |
|---|---|---|---|---|
| behavioural test (384 items x 18 forwards) | 6 min CPU | **1.1 h CPU** | 1.3 M | ~2-4 min compute; ~$0.4 with setup |
| screen (48 bases x 22 forwards of 4 blocks; masked states reuse the behaviour forwards) | 3 min | **0.3 h CPU** | 0.34 M | <1 min; rides on the behaviour job |
| intervention (384 items x 3 recipients x 18 patched forwards) | 19 min | **3.9 h CPU** | 4.5 M | ~5-15 min; ~$0.5 with setup |

**Notes on the estimates:**
- *CPU is free.* It needs about 5.3 h of wall time for the three fresh runs, plus about 30 min of pilots.
- *A10 figures are extrapolated, not measured.* They assume fp32 at 5-15k tokens/s: the bf16 forward measured
  about 25k tokens/s at 34 TFLOP/s, and fp32 peak is 31 TFLOP/s. All three on one A10 instance would come to
  about $0.7.
- *Storage:*
  - behavioural raw readouts, about 20 MB;
  - screen raw: 172 named columns at reference positions plus layer-15 reduced, in float16, about 150 MB, kept
    out of git if over 100 MB per file (checksummed);
  - patching raw changes, about 30 MB.

**Each part leaves behind:** construct sequences and masks, frozen predictions, raw responses, controls, and
per-base summaries with bootstrap intervals.
