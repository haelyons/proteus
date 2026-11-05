# Ruler v1.2 under challenge: expectations (written 2026-11-05, before any challenge column exists)

**Frozen rule.** `atlas/ruler.py` as of commit aab0c82, `calls(..., rule="v1.2")`, BH q = 0.01. Nothing in it
changes during this work. Scoring, calls and evaluation: `scripts/ruler_challenges.py`, one script, so the same
frozen functions score every set.

**A separate, declared diagnostic (not part of the call): local-sequence baseline v0** (`atlas/seqbaseline.py`).
v1.2's null conditions only on the residue at the sampled position, so it cannot tell a column driven by nearby
sequence from one driven by structure (the earlier hydro_win columns were called structural / context). For
each column called structural or context: an independent-sites model of the +-5 window (11 positions x 20
residues, log-odds against the background, 20 pseudo-counts per position) is fitted on the q99.9-q99.999
reservoir tokens (bins 3-4, de-duplicated by cluster), and a background sample of 200,000 residues is
reweighted by it; the peak sample's primary-descriptor counts are tested against that reweighted expectation
(Monte Carlo G, 2,000 draws). Fitting on bins 3-4 and testing on the peak (a disjoint sample) is the
cross-fit. Output: "structure beyond the +-5 window" p-value. Known limits stated now: an independent-sites
model cannot represent a conjunction of residues exactly, and a wider window absorbs more genuine structure.

**Error measurement for the whole procedure** (selection by the atlas, reservoir sampling, de-duplication,
scoring, BH across columns, calls). BH makes a null column's chance of being called depend on how many real
signals share the batch, so error rates are reported at stated mixes: all null, ~40% signal (the challenge
set as built), ~80% signal (null columns subsampled), each with Clopper-Pearson 95% intervals:
- false-call rate = null columns given any association call / null columns;
- false discovery proportion = false calls / all calls;
- power per type and SNR = expected call / columns.

## Challenge set A (fp32; 60,000 non-held-out proteins of the descriptor subset, cropped to 1,022)

| challenge | columns | expected v1.2 call (alpha >= 1, 12 per type) | baseline v0 |
|---|---|---|---|
| motif_nxst: signal on N of N-x-[ST] (x != P), sequence only | 15 (alpha 0.5-4 x 3) | composition association (residue mode N) in >= 9/12; structural or context in <= 6/12 | where structural is called, explained by the window (p > 0.01) in >= 80% |
| motif_gxgxxg: signal on the first G of G-x-G-x-x-G, sequence only | 15 | structural or context in >= 8/12, residue mode G | explained in 4-10 of the structural calls (the conjunction of three G's is only approximated by independent sites) |
| conj_exposed_strand: strand AND exposed tercile | 15 | structural 12/12; primary state5 (strand) in 6-12/12, else burial3 (exposed); the call never names both; in the stored scores the non-primary descriptor is also significant (p < 0.01) in >= 10/12 | not explained (p < 0.01) in >= 10/12 |
| balanced_xor: helix-buried and loop-exposed raised, helix-exposed and loop-buried lowered, state5 and burial3 marginals of the selected set equal to the background (checked on the planted mask, within +-0.01) | 15 | "no association found" in >= 8/12; any detection comes through residue composition or rama4 (second-order), never as an interaction | - |
| hydro_win (as before, for the baseline) | 6 (alpha 1, 3 x 3) | structural or context 6/6 | explained in >= 4/6 |
| helix (positive control for the baseline) | 6 | structural 6/6 | not explained in 6/6 |

Correlated noise (three models, each with 30 null columns and helix / trp / ordered at alpha 1 and 3, 3 each):
protein random effect (rho 0.3), UniRef50-cluster random effect (rho 0.3), smooth noise along the sequence
(white noise filtered over ~10 residues, renormalised).

| quantity | predicted |
|---|---|
| null false-call rate, each model (30 columns) | <= 2/30; pooled over the three <= 3/90 |
| median clusters at the peak (iid ~60) | protein and cluster effects: 20-50; smooth: 45-62 |
| power at alpha 3 (helix, trp, ordered; 9 per model) | >= 8/9 per model |
| power at alpha 1 | at most half lost relative to iid |

Mix columns: 150 iid null columns and the nine original types at alpha 1, 2, 3 (2 each), so that BH sees a
realistic mix. iid null false-call rate <= 1% (upper 95% bound <= 3%); false discovery proportion <= 2%.

## Sets B, C, D: the same columns and noise, transformed

| set | transformation | predicted |
|---|---|---|
| B | z quantised to bf16 before the atlas | call class agrees with A in >= 90% of columns at alpha >= 1; null false-call rate unchanged (<= 2%) |
| C | protein ids permuted (token ids, hashes, tie-breaks and reservoir samples all change) | agreement with A >= 85% at alpha >= 1 (a different sample; disagreements at the detection margin); null false calls <= 2% |
| D | 5% of proteins duplicated with identical noise, same cluster (a model's identical response to identical sequence) | agreement with A >= 85%; null false calls <= 2%; de-duplication removes the copies from every scored sample |

## What will count as a necessary correction, and what as a documented limitation (decided now)

- **Necessary correction** (v1.2 is not used on real columns until fixed and re-challenged): error control
  fails, i.e. the false-call rate among null columns has a 95% interval entirely above 2% under any noise model
  or transformation at the ~40% mix; or a call asserts something false in its own terms (e.g. "composition
  association" for a column whose residues are unrelated to the signal, or "structural association" for a
  null column more often than the error rate allows).
- **Documented limitation** (proceed, and carry it into every real-column report): the claim is true but
  incomplete or needs the baseline to be interpreted, e.g. a conjunction reported through one descriptor; an
  interaction "not detected by this ruler" (never read as "no coherent computation"); a sequence-only column
  called structural (the association is real; "structure beyond the window" is the diagnostic that must
  accompany it); calls that change under bf16 or relabelling only at the detection margin.
