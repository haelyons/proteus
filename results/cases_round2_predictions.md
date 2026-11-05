# Round-2 cases at 8M: what was inspected, proposed computations, frozen predictions

Selection: `real_units_20261105/selection.json` (frozen rule in `cases_round2_plan.md` section 1, seed 20261105,
committed before any example was looked at). Calls: ruler v1.2 over the trained 8M atlas
(`real_units_20261105/calls_pre.json.gz`). The window baseline is not yet repaired, so every structural or context
call below carries **sequence-only not excluded**.

Protein sets:
- *discovery*: non-held-out, seed 2, 600 proteins;
- *fresh*: held-out (created after 2021-11-17) with descriptors, excluding the first pair experiment's 400, seed 1,
  600 proteins;
- for the family test, *every* held-out protein (10,010).

Thresholds are each unit's discovery q99.9 and are never re-estimated on fresh proteins.

Scripts:
- `case_units.py discover` and `case_inspect.py` (atlas and discovery material);
- `case_tests.py <test> --set discovery`, the smoke run; its results are listed below as already inspected;
- then `--set fresh`, the confirmatory run;
- `case_units.py confirm` for the standard readouts (context share at fresh top positions, and silencing at the
  masked position against a write-matched control). The prediction JSONs are in `cases_round2_20261105/pred/`.

## Already inspected, per unit (discovery material only)

### (5,1181) random pick, structural

- **Call:** structural, primary state5, modes loop + helix, persistence to q99.
- **Top-64:** V 23, S 23 in 64 proteins. Peak reservoir: L 78, A 43, V 28.
- **Discovery top positions** (216 above q99.9 = 3.07, in 54 proteins; none within 3 of a terminus):
  - 81% helix (base 39%), 71% buried tercile (base 28%);
  - +-3 enriched for F, C, M, W, I, V, A; depleted for E, K, R, P;
  - **49.5% in Swiss-Prot TRANSMEM segments (base 3.8%)**; 7% in radical SAM cores.
- **Silencing at 54 masked top positions** (one per protein):
  - lowers log p of R, T, V and raises the other 17;
  - mean |change| 0.179 against 0.169 for the write-matched control;
  - median change in log p(true) -0.051 (control -0.033).

### (3,1238) random pick, structural

- **Call:** structural, primary state5, modes loop + strand, peak only.
- **Top-64:** 56 proteins, almost all small heat-shock proteins (sHSP / alpha-crystallin domain): HSPB1-like
  "VSEIRQTADRWR[V]SLDVNHF", IbpA/IbpB "YNIEKTDDNHYR[I]SLALAGF", plant sHSP "YVFAVDMPGIKG[D]EIQV",
  Hsp16.3 "RLEDEMKEGRYE[V]RAELPG", alpha-crystallin. Peak reservoir: many sHSPs, plus radical SAM CxxxCxxC sites.
- **Discovery top positions:** 216 above q99.9 (1.79) sit in only **14 proteins**.
  - **Two sHSPs supply 203:** Q57I25 (IbpA-like, 94 of 142 positions, starting at residue 1) and P29779 (109 of 166).
  - The other 12 proteins give one or two sites each: radical SAM "CxxxCxxC", "LAGGxGxRL" nucleotidyltransferase.
  - 43% strand (base 17%); 79% in a Swiss-Prot DOMAIN; neighbours His-enriched.
- **Silencing at 13 masked positions:** lowers P, T, S, K, D and raises M, L, F, I, V; mean |change| 0.078
  against 0.032 for the control.

### (0,767) random pick, composition

- **Call:** composition, residue modes P, H.
- **Top-64:** all **X** (unknown residue), in 21 proteins, inside runs of X. Peak reservoir: X 169, P 72, H 15.
  Weaker bins: P about 80%, H about 20%.
- **Discovery top positions** (216 above q99.9 = 0.275): centre P 168, H 46; W, Y and D enriched at -1 to -3;
  63% loop.
- **Silencing:** mean |change| 0.002 against 0.0004 for the control; lowers K, R, E slightly.

### (0,789) random pick, context

- **Call:** context, residue mode C.
- **Top-64:** C 63 of 64, in 48 proteins; e.g. repeats "DTTPPTVN[C]" in a Dictyostelium-like S/T-rich protein.
  Peak reservoir: C 69, G 43, A 23, T 20.
- **Discovery top positions** (216 above q99.9 = 0.47, 23 proteins):
  - centre C 44, S 51;
  - +-3 strongly enriched for S and T (log2 +2 to +2.6 at every offset);
  - **75% disordered (base 9%)**, 87% without a pLDDT >= 70 burial value.
- **Silencing:** mean |change| 0.0005 against 0.0009 for the control (no effect beyond control).

### (5,668) random pick, no association found by this procedure

- **Call:** none (burial3 p 0.07; state5 p 0.59).
- **Top-64:** 35 proteins, from high-GC organisms (Streptomyces/Mycobacterium-type, Halobacterium Q9H.../B0R...).
  Windows are rich in A/R/G/P/D/E and poor in K/N/I/F/Y.
- **Discovery top positions:** 216 above q99.9 (1.44) in **12 proteins**; A6WE21 supplies 100 of 223 positions,
  P22409 82 of 621. The relative position is flat along the protein; there is no structural enrichment.
- **Silencing:**
  - lowers Q, H, E, L and raises T, W, M, I;
  - mean |change| 0.060 against 0.019 for the control;
  - median change in log p(true) +0.005 (control -0.012).

### (1,746) deliberate: "not detected", with the strongest next-layer reader

- **Selection:** max cosine 0.321 to layer-2 unit 1006; -0.353 to layer-2 unit 455.
- **Call:** none.
- **Top-64:** only 4 proteins, e.g. P04929 residues 1-12 "MFTSLKKVATFSFLVW…" and Dictyostelium Q54DA1 / Q54D75.
- **Discovery top positions** (216 above q99.9 = 1.19, 40 proteins):
  - **105 of 216 in the first tenth of the protein**;
  - **21% in Swiss-Prot SIGNAL (base 0.4%)**, 9% PROPEP (base 0.4%), 17% TRANSMEM (base 3.8%);
  - examples: "MFSFVDSRTLVLFAATQVILLAV", "MMRKLAILSVSSFLFV", "MKSVRYLIGIFAFIACYYLL".
- **Silencing at the masked position:** mean |change| 0.0063 against 0.0068 for the control (none beyond control).

## (5,668): frozen 2026-11-05 ~18:30 UTC, before any fresh protein was scored for it

**Discovery run of `case_tests.py gc` (already inspected):**
- Spearman(per-protein mean, GARP - FYMINK) 0.568;
- eta2: protein 0.601, centre residue 0.015;
- composition shuffles of the top-20 proteins retain 0.553 of their elevation above the mean of protein means
  (-1.68);
- chimeras (150 + 150 residues, both orders): each half moves a median 0.49 of the gap toward its partner
  (IQR 0.34-0.61), n 80.

**Proposed computation:** a whole-protein composition signal. Every position carries an estimate of the protein's
overall amino-acid makeup along a GC-proteome axis: rich in G/A/R/P (and D/E in halophiles), poor in F/Y/M/I/N/K.
About half of it is composition, which survives shuffling. The rest depends on order, unidentified.

**Alternatives:**
- *taxon/family recognition from specific sequence signatures* predicts shuffle retention near 0;
- *local composition (a window)* predicts chimera shifts near 0 and a large centre-residue share.

**Predictions on fresh proteins:**
- Spearman >= 0.45;
- eta2 protein >= 0.45; eta2 centre residue <= 0.05;
- shuffle retention median >= 0.35 (taxon/family alternative: <= 0.15);
- chimera shift median between 0.25 and 0.75 (local alternative: <= 0.1).

**Standard readouts** (`pred/5_668.json`):
- the context residues L A P S G R at fresh top positions above base (weak; the call found no association);
- silencing at the masked position lowers log p of Q, H, E beyond the write-matched control (the discovery
  direction);
- change in log p(true): no prediction (discovery: unit +0.005, control -0.012).
