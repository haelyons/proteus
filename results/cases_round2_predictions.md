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

## (5,668): frozen 2026-11-05 ~18:10 UTC (commit dafee6e), before any fresh protein was scored for it

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

## (5,1181), (3,1238), (1,746), (0,767): frozen 2026-11-05 18:19 UTC (commit 980b48e), before any fresh protein was scored for them

### (5,1181)

**My first explanation failed on discovery proteins (kept as an unsuccessful explanation).** I guessed "a
hydrophobic segment, read from sequence". The discovery run of `case_tests.py tm` contradicts it:
- Spearman(h, KD19) is 0.132;
- signal peptides never reach the threshold (rate 0, median -0.66);
- inside TM segments the median sits at -0.13 to -0.20 in every KD19 bin, against -1.11 / -0.91 / -0.70 / -0.16
  for buried helices outside TM.

**Discovery material behind the revised proposal:**
- **Distance to a TM segment:**
  - in TM: median -0.15 (rate 1.29%);
  - <= 10 residues away: -0.45 (0.71%);
  - > 10 residues away: -1.14;
  - proteins without TM: -1.26 (0.03%).
- **Buried helices outside TM:** -0.49 (rate 1.4%) in membrane proteins against -1.10 (0.09%) in soluble ones.
- **Protein mean, TM proteins vs others:** AUROC 0.855.
- **40 TM segments:** native median -0.22; inserted into soluble hosts -0.91 (hosts' own median -1.28), i.e.
  retention 0.40; shuffled in place -0.31, retention 0.98.

**Proposed computation:** a membrane-segment context signal. The unit is raised over TM segments and their
flanks (+-10), and further raised within membrane proteins. It reads the segment's composition (insensitive to
order) plus context from the rest of the protein: moved into a soluble protein, a segment keeps only 40%. Its top
values are buried helices of membrane proteins. It does not read local hydrophobicity as such.

**Alternatives:**
- *local hydrophobic segment* predicts Spearman >= 0.4 and insertion retention >= 0.8;
- *buried helix (structural)* predicts buried helices outside TM equal in membrane and soluble proteins
  (difference < 0.2).

**Fresh predictions** (`case_tests.py tm --set fresh`):
- median(TM) - median(no-TM protein) >= 0.7, with the ordering TM > <= 10 > > 10 >= no-TM;
- buried helix outside TM, membrane minus soluble protein >= 0.4;
- AUROC (protein mean) >= 0.75;
- Spearman(h, KD19) <= 0.3;
- in every KD19 bin with >= 20 positions of each, median(TM) >= median(buried helix not TM) + 0.5;
- TM insertion retention between 0.15 and 0.65; shuffle-in-place retention >= 0.7.

**Standard readouts** (`pred/5_1181.json`):
- helix share at fresh top positions above base;
- silencing at the masked position lowers log p of R, T, V beyond the write-matched control. Discovery showed a
  clear direction, but the overall |change| was close to the control's.

### (3,1238)

**Discovery smoke** (`case_tests.py shsp --set discovery`, 3,000 random non-held-out proteins).
- **Amendment after the smoke:** a family hit now also needs the alignment to cover >= 0.4 of the seed. With e <=
  1e-3 alone, 6 hits included A4SG66 and P85978 (seed coverage 0.39 / 0.17, unit fraction 0).
- With the amendment, 4 hits remain: Q943Q3 (identity 0.40) fraction 0.727; P35385 (0.40) 0.509; Q5RAB0
  (1.00 to HSPB8) 0.26; and P35968 (KDR, identity 0.28, coverage 0.43, fraction 0.01), which is probably
  spurious, and is kept.
- **Non-family:** median 0, q95 0. Five reach 10-13% (P84202, Q37369, Q37596, Q9G1N5, P29881).
- **Broadcast:** in the two informative members, shuffling the C-terminal 60% abolishes the N-terminal 40%'s
  response (0.27 -> 0; 0.88 -> 0). Shuffling the N-terminal 40% leaves the C-terminal 60% (median ratio 0.94).
- **Transplants of +-10 windows:** 0 / 30 above the threshold, median -0.2.
- **Composition shuffles:** fraction 0.

**Fresh material:** only 1 held-out protein is an sHSP. The family set is therefore every Swiss-Prot sHSP never
inspected for this unit: not in its atlas top-64 or any reservoir bin, not among the 600 discovery proteins or the
smoke's 3,000, and not a seed. Non-family proteins are the held-out proteins without a hit.

**Proposed computation:** recognition of the sHSP (alpha-crystallin domain) family. The intact domain, more than
a +-10 window, drives a response that is broadcast across the protein, including the N-terminal region, which
responds only while the domain is intact.

**Alternatives:**
- *local motif* predicts transplants >= 50% above the threshold;
- *fold (structural)* predicts CS-domain proteins (the ACD's structural analogues) respond;
- *close relatives only* predicts members below 0.35 identity to every seed do not respond.

**Fresh predictions:**
- family median fraction >= 0.2; non-family median 0 and q95 <= 0.01; AUROC >= 0.9;
- CS-domain median fraction <= 0.01;
- members below 0.35 identity: median fraction >= 0.05, lower than members at >= 0.5 identity (weakly held: the
  smoke had no genuine low-identity member);
- broadcast: N40 with C60 shuffled / N40 native median <= 0.3; C60 with N40 shuffled / C60 native >= 0.6;
- transplants above threshold <= 0.1; composition shuffles median 0, max <= 0.05.

**Standard readouts** (`pred/3_1238.json`): strand share at fresh top positions above base; silencing lowers log p
of P, T, S beyond the control.

### (1,746)

**Discovery run of `case_tests.py nterm`:**
- **Rate above threshold, by segment:**
  - SIGNAL 5.0% (median -0.19); PROPEP 2.4%;
  - TM starting before residue 60: 0.62% (median 0.25); internal TM: 0.40% (median 0.14);
  - other positions in the first 30: 0.17%; all other positions: 0.05%.
- Spearman with KD19: 0.19.
- **Moves:**
  - signal peptides moved to mid-protein: 0.069 -> 0.0 (n 30);
  - internal TM segments moved to the N-terminus: 0.008 -> 0.0 (n 40).
- **Readers** (silencing at 73 top positions, no mask):
  - layer-2 unit 1006: -0.431 (CI -0.450 to -0.412), against +0.013 for the write-matched control;
  - layer-2 unit 455: +0.359 (CI 0.344 to 0.373), against -0.015;
  - for scale, their q50-to-q90 spans are 1.44 and 0.76;
  - the readers' own atlas top-64 are not signal peptides: strand Leu contexts for 1006, Gly contexts for 455.
- **Downstream** (silencing over the signal peptide +10 with 15% of its positions masked; 31 proteins):
  change in log p(true) +5e-5 (CI -0.0024 to 0.0024), against +2e-4 for the control.

**Proposed computation:** a signal-peptide detector. High values need a signal-peptide-type segment at the
N-terminus: neither an N-terminal TM segment nor a displaced signal peptide is enough. Its output is a strong,
opposite-signed input to layer-2 units 1006 and 455. No effect on masked prediction inside the signal peptide is
detected.

**Alternatives:**
- *hydrophobic segment anywhere* predicts SIGNAL and TM rates comparable, and neutral moves;
- *N-terminal position only* predicts the first-30 rate comparable to SIGNAL.

**Fresh predictions:**
- SIGNAL rate >= 10x the base rate and >= 3x each TM category's rate;
- first-30 (other) rate <= SIGNAL / 5;
- moved signal peptides lose >= 80% of their share; TM segments moved to the N-terminus stay <= 0.02;
- readers: 1006 down, 455 up, CIs excluding the control's, |unit mean| >= 5x |control mean|;
- downstream: no effect beyond the control (a predicted null).

**Standard readouts** (`pred/1_746.json`): hydrophobic share at fresh top positions above base; silencing at the
masked position: no effect beyond the control on H/W/Y (discovery: unit 0.0063, control 0.0068).

### (0,767)

**Discovery run of `case_tests.py tok`:**
- eta2 of the centre residue 0.52; rate P 1.6%, H 0.9%, every other residue <= 0.02%;
- no X in the discovery proteins (the atlas top-64 are X);
- +-3 ridge, in sample: R^2 0.39 (P) and 0.41 (H). W at -3 / -2 / +1 / +3 and D at -1 raise the value; K / R at
  +-1 lower it.

**Proposed computation:** a fixed readout of the centre token (X > P > H, far above the rest) plus an
approximately additive neighbour term through layer-0 attention.

**Alternative:** token only, which predicts ridge R^2 <= 0.05 on fresh proteins.

**Fresh predictions:**
- eta2 centre >= 0.4;
- ridge R^2 (fitted on discovery, applied to fresh) >= 0.25 for both P and H;
- P and H the two highest rates among the 20 standard residues;
- X, if present, at a rate >= 0.5.

**Standard readouts** (`pred/0_767.json`): P/H share at top positions; silencing lowers log p of K, R, E beyond the
control (discovery unit 0.002 against 0.0004).

## (0,789): frozen 2026-11-05 ~18:30 UTC (see the commit), before any fresh protein was scored for it

**My first explanation failed on discovery proteins (kept as an unsuccessful explanation).** I guessed "C or S in
an S/T-rich disordered context". The discovery runs of `case_tests.py tok` contradict it:
- Spearman(h, S/T count in +-3) is 0.088 for C and 0.214 for S;
- with the +-3 neighbours set to A, 59% of top C positions stay above the threshold (median 0.55);
- in S/T-rich windows, disordered C has median -0.25 and ordered C -0.50, but n is only 15 / 9.

**Discovery material behind the revised proposal:**
- eta2 of the centre residue 0.084 (C is rare); rate C 1.6%, S 0.36%, T 0.32%;
- C ridge, in sample: R^2 0.44, with large **negative** weights for C at -3 / +3 / -2 (-1.56 / -1.40 / -0.65);
- C with no other C in +-3: median -0.71; with at least one: -2.02 (n 2,326 / 470);
- replacing the neighbouring C by A raises the value by a median **+1.08** (n 80).

**Proposed computation:** a lone-cysteine readout. The centre C token raises the unit, and every other C within
+-3 strongly lowers it, so it separates isolated cysteines from Cys-rich motifs (CxxC, CC, CxC). S/T neighbours add
a little for S.

**Alternative:** the S/T-disorder context, which predicts Spearman(C) >= 0.3 and the +-3 -> A share below 0.2.

**Fresh predictions:**
- C has the highest rate of the 20 residues;
- C ridge R^2 (fitted on discovery) >= 0.3 on fresh proteins;
- lone minus clustered C median >= 0.8;
- neighbouring C -> A raises the value by a median >= 0.6;
- Spearman(C, S/T count) <= 0.2; the +-3 -> A share stays >= 0.4.

**Standard readouts** (`pred/0_789.json`): C share at top positions above base; silencing at the masked position has
no effect beyond the control (discovery: unit 0.0005, control 0.0009).
