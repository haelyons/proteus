# 650M layer 15 unit 2087: when the input carries repeat information, does changing this signal alter how the model uses it?

**Status, 2026-11-06 (first written):**
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


## Discovery pilot (2026-11-06; already inspected, before the freeze below)

- **Material:** `ubq650_use_20261106/constructs_discovery.json`: 8 non-held-out bases (seed 2), 4 targets each.
- **Raw outputs and summaries:** `discovery_pilot/` (`behaviour.pt`, `screen.pt`, `intervention.pt`, `items.csv`,
  `intervention_items.csv`, `summary.json`). Run on CPU, fp32.
- **Smokes before it:** one base, one target (Q4J8E4), all parts, twice, plus the job script end to end on CPU.

**Run incidents:**
- The first pilot's intervention file failed to save: the disk filled because, on CPU,
  `pre15[:, 2087].float().cpu()` stays a view, so `torch.save` wrote the whole layer-15 storage per record.
- Fixed with `.clone()`. Behaviour and screen files were re-saved compactly (values unchanged), and the
  intervention part was re-run.

**Behaviour** (per-base means; 95% bootstrap CI over the 8 bases; all 8 bases share a sign unless noted):

| contrast | mean (CI) |
|---|---|
| agree: log p(t) with copy minus with scramble | +1.43 (0.75 to 2.09) |
| disagree: shift in [log p(v) - log p(t)] | +9.11 (7.93 to 10.35) |
| disagree: log p(t) | -4.09 (-4.99 to -3.20) |
| masked-all: corresponding position masked in both blocks | -0.49 (-0.83 to -0.20) |
| scrambled second block vs R alone | +0.002 (-0.10 to 0.13; 3 / 8 positive) |
| 4 blocks, c = 3 vs c = 0 | +1.60 (1.04 to 2.16) |

- **Confidence:**
  - p(t) with a scramble 0.43 (median 0.22); with a copy 0.988;
  - entropy over the 20 residues 1.62 vs 0.08 nats.
  - The agree recipient is near ceiling. The disagree contrast is not.
- **2087 pre-GELU:**
  - at the masked target: -1.44 with copy, -2.33 with scramble;
  - at the visible counterpart: -2.01 vs -2.34;
  - reference-block mean: -0.98 vs -2.27;
  - masked-all target -2.01; c = 3 target -1.20 vs c = 0 -2.21.

**Screen** (4 blocks, constant length, unmasked, R in slot 1):

| copies | 2087 reference-block mean | share of reference positions > q99 (0.57) | share > q99.9 (2.19) | mean maximum |
|---|---|---|---|---|
| c = 0 | -2.14 | 0.2% | 0% | -0.25 |
| c = 1 | -0.73 | 33% | 4% | 2.98 |
| c = 2 | -0.73 | 40% | 8% | 3.51 |
| c = 3 | -0.71 | 43% | 12% | 3.80 |

- Above q99.9 at c = 3, by residue: H 0.75 (n 12), C 0.75 (4), P 0.62 (34), S 0.39 (52).
- Masked c = 1 / 2 reference-block means: -0.74 / -0.74.
- **Columns** (pole-adjusted z, c = 3 vs c = 0):
  - the 40 ubiquitin-selected columns: median 0.41, 9 with z > 2, 4 with z > 5 (largest 32/1659, 10/3519, 9/2946,
    20/2491);
  - the 132 comparison columns: median 0.01, 1 with z > 2.
- **Layer 15:** 2087 has z 2.05, rank 8 of 5,120. The quantiles of all units are median -0.23, q99 1.22,
  q99.9 2.22.
- **Selected group** (rule in `ubq650_use.py select_group`): 2087, 4330, 1493, 1284, 1659, 1526, 3318, 4544, 4133,
  2481, 1444, 1148, 2600, 3743, 4759, 4449, 4645, 3775, 4531, 704. Unit 4330 was the write-matched control of the
  earlier silencing test.

**Intervention.** f = fraction of the contrast removed (ratio of sums over bases, CI over bases).

| recipient | 2087, alpha = 1, all positions | 20-unit group, alpha = 1 |
|---|---|---|
| agree | -0.0000 (-0.0001 to 0.0001) | -0.0003 (-0.0009 to 0.0001) |
| disagree (copying preference) | -0.0008 (-0.0023 to 0.0004) | -0.0033 (-0.0090 to 0.0012) |
| 4 blocks, c = 3 | -0.0000 | +0.0001 |

- Unit and random-direction controls are of the same order: |f| <= 0.0011.
- KL at the target is <= 6e-4 throughout.
- 2087's own patch is graded but tiny: disagree d log p(t) -0.0009 / -0.0020 / -0.0031 / -0.0042 at alpha 0.25 /
  0.5 / 0.75 / 1.
- **Control failure:** random 20-unit groups reached only 0.37-0.57 of the group's combined residual change, so
  the group comparison was under-matched.

## Amendments after the pilot, before any fresh base is scored

1. **Bug fix:** `.clone()` on saved per-position vectors. Values are unchanged.
2. **Group controls:**
   - random same-size groups are drawn from all non-group units, then from the top 2,000 / 1,000 / 500 / 200
     non-group units by single-unit change norm, until 4 lie within +-10% of the group's combined norm;
   - the pool and ratio are recorded per item;
   - plus 2 group-level random directions with the group's per-position combined-change norm.
   - Smoke: ratios 0.92-1.05 from the top-500 / top-200 pools.
3. **Secondary upper bound:** the whole layer-15 FFN output (all 5,120 units) is patched toward the donor
   (alpha 0.5, 1), against 2 random directions with its per-position norm.
4. **The discovery intervention is re-run on the GPU** with the amended controls, recorded as discovery, in the
   same job before the fresh run (`run.sh` stage 1b). Predictions below cover it too.

## FROZEN 2026-11-06, before any fresh base is scored

**Inputs:**
- `constructs_fresh.json`: sha256 4582a12b...ebadd; 48 held-out bases (seed 1), 8 targets each; disagreeing
  residues balanced 19-20 per amino acid.
- `group_frozen.json`: sha256 6cbe15e3...2eb5; the pilot's group.

**Run setup:** A10, fp32, TF32 off. Stage 1 compares GPU with CPU on the smoke constructs.

**Inference:** per-base means, 95% bootstrap over the 48 bases (2,000 draws), with the number of bases sharing the
sign. Raw per-item values are kept in `items.csv`, `intervention_items.csv` and the `.pt` files.

**Behaviour.** Decision rule with delta_b = 0.1 nats:
- *effect:* the interval excludes 0 and lies beyond 0.1 in the stated direction;
- *excluded:* the interval lies within +-0.1;
- *inconclusive:* otherwise.

| contrast | prediction |
|---|---|
| agree | effect, +; mean 0.8-2.5 nats |
| disagree preference shift | effect, +; mean >= 5 nats |
| disagree log p(t) | effect, - |
| masked-all | effect, - (weakly held: pilot -0.49) |
| scramble block vs R alone | excluded (scramble neutral) |
| 4-block c = 3 vs c = 0 | effect, + |
| order RX vs XR | agree means within 0.3 nats |

Confidence and unit activity (recorded, with ranges predicted):
- p(t) with copy >= 0.95; with scramble 0.25-0.6;
- 2087 at the masked target: below its q90 (-0.96) on average with copy; above the scramble value by 0.5-1.5;
- reference-block mean: copy minus scramble >= 0.8.

**Screen.** For 2087:
- share of reference positions > q99: <= 2% at c = 0 and >= 20% at each of c = 1, 2, 3;
- reference-block mean at c >= 1 minus c = 0: >= 1.0;
- H, C or P among the three residues most often above q99.9 at c = 3;
- masked vs unmasked reference-block mean: within 0.1.

Columns: the ubiquitin-selected set has more columns with pole-adjusted z > 2 than the comparison set, by count
and by proportion (pilot 9 / 40 vs 1 / 132), and a higher median z.

**Intervention.** For each recipient (agree, disagree, 4-block c = 3), with meaningful effect size delta_i = 0.02,
i.e. 2% of that recipient's behavioural contrast:
- *effect:* the lower bound of f exceeds 0.02, and the point estimate exceeds the upper bound of each pooled
  matched-control family (unit controls, random directions, matched groups / group random directions);
- *excluded:* the interval for f lies within (-0.02, 0.02);
- *inconclusive:* otherwise.

| patch | prediction | conclusion it addresses |
|---|---|---|
| 2087, alpha = 1, all positions | **excluded** in all three recipients | "accounts for part of the repeat effect" |
| 2087, reference block only / other blocks only | excluded | |
| 20-unit group, alpha = 1 | **excluded** in all three | |
| 2087 and group: KL at the target | no larger than the pooled matched controls' (upper CI) | "moves predictions" |
| 2087 and group: agree d log p(t) | within +-0.02 nats | "helps prediction" |
| 2087 and group: disagree preference | f within +-0.02 | "reduces copying" |
| whole layer-15 FFN, alpha = 1 (secondary) | f between 0 and 0.2 in agree and disagree, not excluded (weakly held) | upper bound for layer 15's FFN |

**Wording fixed now:**
- *If 2087 is excluded:* "Replacing 2087's output with its scramble-construct values, at every position,
  changes the model's use of the copy by less than 2% of the behavioural contrast (95% CI over 48 fresh bases).
  This closes the route that this signal, alone or with its 19 strongest layer-15 co-responders, carries the
  repeat effect in these constructs. It does not settle other roles: different constructs, natural repeats, or
  effects at positions not read out here."
- *If even the whole layer-15 FFN is excluded:* the repeat use is carried elsewhere (attention, or FFNs in
  other layers). Readers stay deferred until a contrast worth explaining exists at some layer.
