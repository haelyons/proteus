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
