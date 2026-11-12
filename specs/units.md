# Spec: reading units by intervention at 8M (shared intervention module and the unit study)

Status: frozen at the commit that adds this file. A new line, separate from the comparison:
columns are read one by one and each reading is tested by intervention on held-out proteins.
Neurons and random directions are two populations put through the same steps. A direction is
never paired with a neuron; every control for a column is drawn from the column's own
population. Not in scope: attribution, attention sites, every-unit runs, 650M.

## 1. `atlas/intervene.py` (one module, from `scripts/unit_writes.py`, `scripts/case_units.py`,
`scripts/travel_v2.py`)

```
class Intervener:
    def __init__(self, model="8M", device="cpu", dtype=torch.float32)
    def run(self, ids: Tensor[1, T], edit: Edit | None = None, capture=False) -> Run
    def replace(self, ids, column: Column, positions: list[int], values: list[float],
                k_later: int = 10, baseline: Run | None = None) -> Result
    def head(self, x_final: Tensor[n, d_model]) -> Tensor[n, 20]   # emb_layer_norm_after, then lm_head; log p of the 20 residues
    def column_values(self, run: Run, column: Column) -> Tensor[n]  # h . r at every residue position
Column = (layer: int, r: Tensor[d_ff])        # unit vector; neuron u is the one-hot e_u
Edit   = (layer, r, positions, values)
Run:    lp: Tensor[n, 20] (float64 log p of the 20 residues in AA order "ACDEFGHIKLMNPQRSTVWY", from
        log_softmax over the full vocabulary), lp_full: Tensor[T, V] float64, post: {l: Tensor[n, d_ff]}
        if capture (post-GELU, residues only), x_final: Tensor[n, d_model] (residual before
        emb_layer_norm_after, residues only)
Result: dlp: Tensor[n, 20] (edited minus baseline), later: [(layer, unit, change)] (k_later largest),
        later_share: float (their share of the total change), write: Tensor[len(positions)]
        (|v - h.r| * ||W_out r||, per position)
```

Positions are residue indices (0-based, excluding BOS/EOS). Hook: the output of
`layer.intermediate` (post-GELU), as the three scripts. **Edit rule**, for one column
`(l, r)` at positions `P` with values `v`: with `a = h[P] @ r` (fp32 matmul), `h[P] <- h[P] + (v - a)[:, None]
* r[None, :]`, applied only at positions where `v != a`. A neuron and the direction equal to its
axis take the same code path. Later change: for each later layer `l' > l` and unit `u'`,
`sum over residues |post_edit - post_base|`; `later` is the `k_later` largest (ties by layer, then
unit), `later_share` is their sum over the sum over all later units (null at the last layer).

**Invariants (blind tests, CPU fp32, ESM-2 8M from the HF cache):**
- Replacing a column's value with itself (`values = a`) leaves `lp_full` and every captured `post`
  bitwise unchanged.
- For neuron u and the direction `r = e_u`, `replace` gives bitwise equal `dlp`, `later` and
  `write`.
- Reproduction: silencing (value 0) the recorded cases of `results/unit_writes_8M_20261105/cases.jsonl`
  gives each case's `dlogp_true`, `kl` and `travel_kl`, and the same for `base_unit`: `dlogp_true`
  within 1e-6 absolute, `kl` and `travel_kl` within 1e-6 absolute or 1e-4 relative (`kl` is over
  the full vocabulary, as there). The recorded noise floor is 0.0 at one thread on the recording
  machine; another platform can differ at the last bit (seen for `randn` plus normalisation), so
  the number of cases equal bitwise is reported, not required. This needs the full Swiss-Prot cache; the test takes its path from
  `PROTEUS_SWISSPROT` and skips when absent. The study job runs it on the box (step 1 below).

## 2. Frozen sample

From the calibration run's atlases (`$CALIB/atlases/atlas_neurons_post.pt` = N,
`atlas_randdir_512_0.pt` = R; `CALIB=$FS/runs/calib-8m-20261112-1018`): per layer, 25 neurons
(`torch.randperm(1280, generator=manual_seed(20261112 * 10 + l))[:25]`) and 25 directions
(`torch.randperm(512, generator=manual_seed(20261112 * 10 + 100 + l))[:25]`), 300 columns. The
directions' vectors are regenerated with `atlas.columns.parse_sources("randdir:512:0", 1280)` in
fp32; their sha256 (as `run_atlas.py` computes it) is recorded next to the calibration's
(`625afb88...`). A difference is recorded, not fatal (it is at the last bit: the box and this
machine already differ).

## 3. Per-column protocol

**Material.** Discovery: the atlas only (top-64 tokens, peak reservoir, edges). Tests: held-out
proteins (`heldout_created_after_2021-11-17`), loaded as `case_units.py confirm` does
(`pair_balance.proteins(a, True, n, seed)` from the descriptor subset), `n = 1000`, `seed = 1`.
One unmasked forward per held-out protein gives every column's values (`column_values`).

**Thresholds** from the atlas edges of the column (positive pole): median = edge q0.5, peak
threshold = edge q0.999, steering value = edge q0.999.

**Positions.** Peak: as `case_units.unit_positions`: held-out positions with value >= q0.999,
margin 3 from either end, at most one per protein (the first), at most 60 in protein order.
Random: 60 (protein, position) pairs with margin 3 from a generator seeded by
`20261112 + 1000 * layer + column index`, excluding the column's peak positions. Every test
forward masks the tested position (`case_units.silence`): the masked baseline is shared by the
target and its controls at that position, and effects are read at the masked position.

**Effect of an edit at a position** = KL(p_base || p_edit) over the 20 residues (log p of the 20,
renormalised over the 20), at the masked position.

a. **Profile** (discovery): the residue at each of the top-64 and peak-reservoir tokens, their
   `ss3` (3 states plus missing) and burial class (tertiles of burial over residues with pLDDT >= 70
   in the descriptor subset's non-held-out proteins, plus "pLDDT < 70"), from the whole-build
   descriptor file. Card: the 64 top tokens with +-3 residues, the three distributions.
b. **Direct write**: at the 60 random positions, `head(x_final + (q0.999 - a) * W_out r)` minus
   `head(x_final)` from the masked baseline: mean change of log p per residue (20 values).
c. **Removal at peak**, at the peak positions, two replacements: median (the column set to its
   median), and resampling (the column set to its value at a held-out token drawn uniformly, three
   draws per position, effects averaged). Controls, each from the column's own population and layer:
   (i) a random other column at the same positions, with its own median; (ii) a random other column
   at the same positions with its change scaled to the same write size, `v' = a' + (v - a) *
   ||W_out r|| / ||W_out r'||`; (iii) the same column at the random positions, same replacement.
   One control column per target per control type, drawn from a generator seeded by
   `20261112 + 7 + 1000 * layer + column index`.
d. **Later units** (median removal at peak, all peak positions summed): the 10 later units
   (neurons of later layers) that change most and `later_share`.
e. **Steering**: the column set to q0.999 at the 60 random positions: mean change of log p per
   residue (20 values).

A column with fewer than 10 peak positions is **insufficient** for checks (1) and (3).

## 4. Checks (mechanical)

Each check compares the column with 20 other columns of its own population and layer, drawn from a
generator seeded by `20261112 + 13 + 1000 * layer + column index` (pass means p < 0.05 by rank).

1. **Profile**: distance = mean of the Jensen-Shannon divergences (base 2) of the three
   distributions (residue, `ss3`, burial) between the column's held-out peak tokens and a discovery
   profile. Pass if the distance to its own profile is smaller than the distance to each of the 20
   others' profiles.
2. **Write predicts steering**: Spearman correlation (average ranks for ties) between a column's
   b-vector and its e-vector. Pass if its own correlation is larger than the correlation between
   each of the 20 others' b-vectors and its e-vector.
3. **Removal at peak exceeds each control**: for median replacement, the 95% bootstrap interval
   (2,000 draws, seeded 0; positions resampled jointly for (i) and (ii), independently for (iii))
   of mean target effect minus mean control effect lies above 0, for all three controls. Also
   reported for resampling replacement (secondary).

### 4b. Amendment (2026-11-12, before any run): frozen helper interface

The first version named no helpers for the check arithmetic, and the blind tests had to assume
names. These are now frozen in `scripts/units_study.py`, and the study computes the checks with them:

```
profile_distance(p, q) -> float        # p, q: [residue(20), ss3(4), burial(4)] probability vectors; mean base-2 JSD
rank_pass(own, controls, higher=False) -> bool   # higher=False: own < every control (check 1); True: own > every control (check 2)
spearman(x, y) -> float                # average ranks for ties
bootstrap_diff(target, control, paired=True, draws=2000, seed=0, level=0.95) -> [lo, hi]
                                       # interval of mean(target) - mean(control); paired=True resamples positions
                                       # jointly (equal lengths required), False independently; check 3 passes a
                                       # control when lo > 0
```

Cards expose `checks` with keys `profile`, `write_steer`, `removal`, each with a boolean `pass`
(null when insufficient); `removal` also holds `resample` with its own `pass`.

Pass rates per population (neurons, directions): passes over sufficient columns, with
Clopper-Pearson 95% intervals; also per layer. A difference between populations is described,
never tested as a pairing.

## 5. Job and outputs (`jobs/units_8m/run.sh`, frozen keys)

Env `JOB OUT FS DEADLINE`, optional `WORKERS` (default `nproc - 2`), `CALIB`. Inputs (read only;
path, size, sha256 into `$OUT/inputs.json`, a missing one fails the job): `$FS/cache/swissprot_2026_03.pt`,
`$FS/runs/atlas-wab-20261105-0140/desc_subset/`, `$FS/descriptors_swissprot_2026_03.pt`,
`lists/swissprot_2026_03_uniref50.tsv.gz`, the two calibration atlases. Steps: `inputs`;
`repro_unit_writes` (section 1, all 900 cases; a failure is recorded and stops the job before the
study); `sample`; `heldout_values` (one forward per held-out protein); `columns` (the 300 columns
across `WORKERS` processes, one card each); `report`. CPU only, fp32, one torch thread per process.

`$OUT/sample.json` (`{"neurons": [[l, u], ...], "directions": [[l, k], ...], "directions_sha256",
"calib_directions_sha256"}`), `$OUT/repro_unit_writes.json` (`{"n_cases", "n_bitwise",
"max_abs_diff": {"dlogp_true", "kl", "travel_kl"}, "ok"}`), `$OUT/cards/<pop>_<l>_<k>.json` and `.md` (one per
column: a–e and the three checks), `$OUT/units.json`:

```
{"populations": {"neurons": {"n", "insufficient",
                             "pass": {"profile": {"k", "n", "rate", "ci"}, "write_steer": {...},
                                      "removal": {...}, "removal_resample": {...}},
                             "by_layer": {"0": {...same...}, ...}},
                 "directions": {...same...}},
 "seeds": {...}, "positions": {"peak_max": 60, "random": 60, "margin": 3}}
```

Worked example (arithmetic of one pass rate): 150 directions, 12 insufficient, 23 of 138 pass the
profile check: `{"k": 23, "n": 138, "rate": 0.16667, "ci": [0.10868, 0.23951]}` (Clopper-Pearson
95%, `atlas.compare.share_ci`).

`SMOKE=1`: CPU, the Swiss-Prot test set, synthetic descriptors and a small atlas built in the job
(`--columns neurons_post,randdir:512:0`, tiny config), 2 columns per population per layer, 40
held-out stand-in proteins, 20 random positions, 4 peak positions minimum; must finish in 30
minutes on four cores.

## Acceptance (blind tests; invariants and arithmetic)

- The three invariants of section 1 (the reproduction one with `PROTEUS_SWISSPROT`).
- The edit leaves every position outside `positions` unchanged in the edited layer's `post`.
- Check arithmetic on hand-built inputs: JSD distances and the rank rule of check 1; Spearman with
  ties of check 2; the bootstrap rule of check 3 (a control equal to the target never passes; a
  target larger than every control at every position passes); the worked example's interval.
- `SMOKE=1` end to end; `units.json` keys; every card lists exactly the three checks; pass counts
  in `units.json` equal the counts over the cards.

## Cost

CPU work: about 600 forwards per column, 300 columns, plus 1,000 held-out forwards and the 900-case
reproduction: about 3 thread-hours. One `gpu_1x_a10` (30 vCPU) in us-east-1 on
`esm-atlas-results`, 28 processes: about 15–30 minutes of compute; cap 1.5 h, at most $1.94.
Launch only with the owner's yes.
