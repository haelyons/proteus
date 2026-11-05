# 650M layer 15 unit 2087 (positive pole): written 2026-11-05 before any probe is run

Selected by the frozen rule (`cases_round2_plan.md` section 1: largest ubiquitin share of the top-64; 1.0).

## Already inspected (discovery material)

- Atlas #2: top-64 values 6.66-6.91, every one at **His68** of a ubiquitin copy, in 8 proteins.
- Peak reservoir (>= q99.999, 64 samples): only 4 ubiquitin tokens; H 31, P 21, Q 5, N 4.
- q99.9-q99.99 bin: S 15, P 14, H 8.
- q90-q99 bin: P 15, L 11, S 9.
- Bin edges (pre-GELU): q50 -2.02, q90 -0.96, q99 0.57, q99.9 2.19, q99.99 3.53, q99.999 4.53.
- Ubiquitin context: H68 sits in beta5 (65-STLHLVLRLRGG-76), part of the L8 / I44 / H68 / V70 hydrophobic
  patch that ubiquitin-binding domains read.

## Competing explanations

- **A, family recognition:** the unit fires maximally at His68 because the whole sequence is ubiquitin-like;
  it needs ubiquitin's global sequence.
- **B, local motif:** a His (or Pro) in a local window like L-H-L-V-L; ubiquitin's H68 window is simply its
  best match.
- **C, structural:** His in a beta-strand at a hydrophobic surface patch, read through context that the model
  infers beyond the local window.

## Probes (650M, fp32, CPU; pre-GELU value of unit 2087 at the residue named)

| probe | A predicts | B predicts | C predicts |
|---|---|---|---|
| human ubiquitin alone, H68 | high (~6.7) | high | high |
| NEDD8 (local 65-71 nearly identical, ~58% identical overall), its H68 | high | high | high |
| ubiquitin residues 61-75 transplanted into 10 unrelated proteins (random positions), at the H | low | retained (>= 50% of the isolated value) | depends on host |
| ubiquitin with residues 1-50 shuffled (51-76 kept), H68 | low | retained (>= 80%) | low (fold destroyed) |
| distal point mutants L8A, I44A, V70A (V70 is local) | L8A, I44A lower | L8A, I44A unchanged, V70A lower | L8A, I44A lower |
| local mutants L67A, L69A | modest | lower | lower |
| Parkin UBL (V-H-I-V at the homologous position) | moderate (family) | moderate (similar window) | moderate |
| SUMO1, RAD23A UBL, ISG15 domains (no His there), max over the sequence | low | low unless an H/P window matches | low |
| 20 composition-preserving shuffles of ubiquitin, max at any H | low | low (window destroyed) | low |

**My prediction:** B. Transplants retain >= 50%, the distal shuffle retains >= 80%, and L8A / I44A change it by
less than 10%. The unit is then a local His-window detector whose best match is ubiquitin's beta5 window: family
"recognition" only in the sense that the family shares that window. If A or C holds instead, the unit reads more
than the local sequence.

## Result of the pre-registered probes (run 17:25 UTC): all three explanations fail the first probe

Isolated human ubiquitin, H68: **-1.34** pre-GELU (between this unit's q50 -2.02 and q90 -0.96). A, B and C all
predicted high (~6.7). NEDD8 H: -1.44. Transplants: median -2.3. Distal shuffles: -1.3 to -1.7. Mutants:
-1.28 to -1.96 (H68A -1.96). Parkin H: -1.54. Composition shuffles: max at His -0.7 to -2.9. The highest value
in isolated ubiquitin is 0.76, at Q41. Values are fp32 on CPU; the atlas used bf16 on GPU, but a gap of 8 units
is not numerics.

## Post-hoc hypothesis D, written after seeing that result and before testing it

The top-64 (6.66-6.91) all come from ubiquitin copies inside polyubiquitin precursors and fusions. D: the unit's
extreme response to H68 needs **ubiquitin repeated in tandem within the same protein**. It would then respond to
"this window has already occurred earlier in the sequence", with ubiquitin as the best-represented case, not to
ubiquitin as such.

| probe | D predicts |
|---|---|
| human UBC (P0CG48, 9 tandem copies): H68 of each copy | copy 1 low (< q90 -0.96); copies 2-9 high (> q99.99, 3.53) |
| UBB (3 copies) | copy 1 low; copies 2-3 high |
| UBA52, RPS27A (one copy fused to a ribosomal protein) | low |
| synthetic ubiquitin + ubiquitin (2 copies) | copy 2 high, copy 1 low |
| synthetic ubiquitin + 40 unrelated residues + ubiquitin | copy 2 high (lower than adjacent copies allowed) |
| synthetic SUMO1 + SUMO1, and an unrelated 80-residue domain twice | at the second copy, the unit rises at some positions to well above its value in the first copy (> q99.9, 2.19, somewhere), if "repeat" rather than "ubiquitin repeat" is what it reads |
| peak-bin (q99.999) samples outside ubiquitin (H 31, P 21 of 64) | mostly from proteins with internal repeats of their window |

If copies 2+ of UBC are not high, D fails too, and the atlas values remain unexplained (numerics or batch
context would then be the next suspects).

## Result for D (run 17:3x UTC) and a dose-response test written before running it

Observed:
- UBC copies 1-9 at H68: 6.67, 6.72, 6.69, 6.67, 6.62, 6.61, 6.60, 6.59, 5.72;
- UBB copies 1-3: 3.82, 3.83, 3.35;
- UBA52 and RPS27A (one copy): -1.40 and -1.38;
- synthetic Ub+Ub: 2.14 / 2.22; Ub+40+Ub: 1.84 / 2.37;
- SUMO1 x2: max 2.58 / 2.43 against 0.97 single; an unrelated 80-mer (lysozyme 19-98) x2: 2.90 / 2.28 against 1.12 single;
- 19 of 61 non-ubiquitin peak samples sit in proteins that repeat their exact +-5 window; others look like
  approximate repeats (C2H2 zinc-finger linkers THQRIHTGEKP; HQCGLP / HRCGLP).

D's repeat dependence holds and generalises beyond ubiquitin. Its direction ("already occurred earlier") is
wrong: copy 1 is as high as later copies, since the model reads both ways.

Revised proposal (D'): the unit's value grows with the number of (near-)identical copies of the local sequence
in the protein; ubiquitin fills its top-64 because polyubiquitin precursors carry the most identical tandem
copies in Swiss-Prot.

Dose-response predictions (synthetic tandem copies, 650M, H68 of copy 1 and the median over copies):

| copies of ubiquitin | 1 | 2 | 3 | 4 | 6 | 9 |
|---|---|---|---|---|---|---|
| D' predicts | ~ -1.3 | ~ 2 | rising | rising | rising | ~ 6.5 (UBC) |

Monotone increase in copy number, copy 1 and later copies alike. For 5 random held-out protein fragments (80
residues, none containing ubiquitin): the maximum over the construct rises from 1 to 2 to 4 copies in >= 4 of 5.
If it does not, D' is wrong.

## Dose-response result (D' predictions above)

Ubiquitin copies 1 / 2 / 3 / 4 / 6 / 9 give H68 values of -1.34 / 2.14 / 3.93 / 5.02 / 6.06 / 6.68: monotone and
saturating, with copy 1 equal to the later copies. Held-out fragments x1 / x2 / x4 rise monotonically in **5 / 5**
(e.g. 0.39 / 3.09 / 4.54). D' holds on fresh sequences.

## Downstream test, written before running it

Does the model use the unit when predicting masked residues in repeats? In Ub x4, mask a random 15% of copy 2's
positions (seeded). Silence unit 2087 (post-GELU = 0) at every position, against a control: another layer-15 unit
whose summed |GELU| x |w_out| over the sequence is closest to 2087's in the same masked forward. Readout: signed
change in log p(true) at the masked positions. The same in a single ubiquitin (unit near its off level).

Predicted:
- in Ub x4, silencing 2087 lowers mean log p(true) at the masked positions, by more than twice the matched
  control's |change| but by < 0.05 nats (one unit of 168,960);
- in a single ubiquitin: no change beyond the control.

If silencing changes nothing beyond the control, the unit reads repeats without the model detectably using it
for masked prediction at this scale. That would be reported as such.
