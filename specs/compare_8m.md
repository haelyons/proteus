# Spec: the 8M comparison (N, R, S1–S3, RS1–RS3) and the H1 verdict

Status: frozen at the commit that adds this file. Implements `results/science_decision_rule.md`
(sealed) with amendment 1. Inputs are the calibration run's outputs
(`calib-8m-20261112-1018`, `results/calib_8m_20261112/`). `atlas/ruler.py` stays byte-identical.
After this job the 8M comparison line stops: nothing at 650M, 3B or with SAE latents follows
without the owner's decision.

## Sources

| Name | Columns | Atlas |
| --- | --- | --- |
| `N` | trained post-GELU neurons | `$CALIB/atlases/atlas_neurons_post.pt` (calibration run) |
| `R` | `randdir:512:0` in the trained network | `$CALIB/atlases/atlas_randdir_512_0.pt` (calibration run) |
| `S1`–`S3` | post-GELU neurons of shuffled control c | built here, `atlas_neurons_post.pt` of run `shuf<c>` |
| `RS1`–`RS3` | `randdir:512:0` in shuffled control c, same forward as `S<c>` | built here, `atlas_randdir_512_0.pt` of run `shuf<c>` |

`CALIB=$FS/runs/calib-8m-20261112-1018`.

## `scripts/compare_report.py`

```
python scripts/compare_report.py --scores SCORES_DIR --summary SUMMARY_JSON [--repro REPRO_JSON]
                                 [--draws 2000] [--seed 0] [--level 0.95] --out COMPARE_JSON
```

Reads `SCORES_DIR/<NAME>.jsonl` for the eight names, `SCORES_DIR/<NAME>.scores.pt` (for peak
log2), and the `summarise` output for pooled v1.2 calls. Writes `COMPARE_JSON` and a Markdown
table beside it (`.md`).

Library functions (in the same file, importable):

```
bootstrap_cliffs(x: list[float], y: list[float], *, draws=2000, seed=0, level=0.95) -> [lo, hi]
h1_verdict(d_NR, p_NR, ci_NR, d_NS: list[float], ci_SRS: list[[lo, hi]]) -> str
```

**Values.** Every comparison uses `primary_jsd` of the non-`insufficient` records of each source,
in record order. Cliff's d and the Mann-Whitney p are `scripts/separation_gate.py`'s
`cliffs_delta` and `mann_whitney` (first argument minus second: d > 0 means the first is higher).

**Bootstrap.** One `torch.Generator().manual_seed(seed)` per comparison. For each of `draws`
draws, in this order: `ix = torch.randint(len(x), (len(x),), generator=g)`, then
`iy = torch.randint(len(y), (len(y),), generator=g)`; the draw's value is
`cliffs_delta([x[i] for i in ix], [y[i] for i in iy])`. The interval is
`torch.quantile(torch.tensor(values, dtype=torch.float64), torch.tensor([(1 - level) / 2,
(1 + level) / 2], dtype=torch.float64))` (linear interpolation), as two Python floats.

**Verdict** (the sealed rule with amendment 1), evaluated in this order:

1. `d_NR <= -0.147 and p_NR < 0.01`: `"strongly against"`.
2. `d_NR < 0.147 or p_NR >= 0.01`: `"against"`.
3. Otherwise, if not every `ci_NR[0] > ci[1]` for `ci` in `ci_SRS` (strict; touching intervals
   do not separate): `"gap not attributable to training"`.
4. Otherwise `d_NR >= 0.5 * mean(d_NS)`: `"supports"`; else `"weak support"`.

### `compare.json` (frozen keys)

```
{"n": {NAME: n_non_insufficient},
 "calibration": {"S1": {"d", "p", "ci"}, "S2": {...}, "S3": {...}},     # d(N, Sc)
 "calibration_passes": bool,                                            # every d >= 0.147 and p < 0.01
 "d_NR": {"d", "p", "ci"},
 "d_SRS": {"S1": {"d", "p", "ci"}, "S2": {...}, "S3": {...}},          # d(Sc, RSc)
 "d_NS_mean": x,
 "h1": VERDICT or null,                                                 # null if calibration fails
 "h1_conditions": {"d_ge_0147": bool, "p_lt_001": bool, "attributable": bool, "half_gap": bool},
 "secondary": {"share_residue_mode": {NAME: {"k", "n", "share", "ci"}},  # separation_gate.stat_summary
               "median_primary_jsd": {NAME: x},
               "peak_log2_median": {NAME: x},                              # as separation_gate
               "pooled_v12_calls": {NAME: {CALL: n}}},                     # one calls() over all eight
 "bootstrap": {"draws", "seed", "level"},
 "repro": {...as REPRO_JSON, or null}}
```

### Worked example (verdict)

`d_NR = 0.30`, `p_NR = 1e-10`, `ci_NR = [0.26, 0.34]`, `d_NS = [0.236, 0.214, 0.198]` (mean
0.216, half 0.108), `ci_SRS = [[-0.05, 0.03], [-0.02, 0.06], [0.00, 0.08]]`: steps 1–2 do not
apply; 0.26 is above 0.03, 0.06 and 0.08; 0.30 >= 0.108: **"supports"**. With
`ci_SRS[2] = [0.20, 0.26]` instead: 0.26 is not above 0.26: **"gap not attributable to
training"**. With `d_NR = 0.15`, `ci_NR = [0.11, 0.19]`, `ci_SRS` all below 0.10, `d_NS` mean
0.40: **"weak support"**. With `d_NR = 0.10`: **"against"**. With `d_NR = -0.20`, `p_NR = 1e-5`:
**"strongly against"**.

Bootstrap example: `x = [1, 2, 3]`, `y = [4, 5]`: every draw gives d = -1, so the interval is
`[-1.0, -1.0]`.

## `jobs/compare_8m/run.sh`

Env `JOB OUT FS DEADLINE`; optional `WORKERS` (default `nproc - 2`), `PER_LAYER` (default 250),
`CALIB` (default as above). Tolerates the inbox runner's files in `OUT`; `SCRATCH` as
`jobs/calib_8m/run.sh`. Non-smoke preflight, GPU record, 1 Hz logger and GPU tests as
`jobs/calib_8m/run.sh`.

1. `inputs`: path, size and sha256 of the N and R atlases, the calibration `pooling/sample.json`
   and `pooling/scores/{trained,shuf1,shuf2,shuf3}.jsonl`, the data and descriptor files, into
   `$OUT/inputs.json`; a missing input fails here.
2. `shuf<c>`, c = 1, 2, 3: the calibration config (`specs/calib.md` section 3, with 3b) plus
   `--control shuffled --control-seed c --columns neurons_post,randdir:512:0
   --source-cap neurons_post=0.1` into `$SCRATCH/shuf<c>`; copy its `run.json` and
   `batches.jsonl` to `$OUT/shuf<c>/`.
3. `sample`: `compare_sources.py sample --atlas N=... R=... S1=... S2=... S3=... RS1=... RS2=...
   RS3=... --per-layer PER_LAYER --seed 0 --out $OUT/sample.json`.
4. `score_<NAME>` for the eight names: `compare_sources.py score` with `--desc` the whole-build
   descriptor file, into `$OUT/scores/`.
5. `repro`: for `N` against the calibration's `trained.jsonl` and `S<c>` against its
   `shuf<c>.jsonl`: matching by `(layer, col)`, the number of records and the number with equal
   `score_sha256` → `$OUT/repro.json` `{NAME: {"n", "identical"}}`. A mismatch is recorded and
   does not stop the job.
6. `summary`: `compare_sources.py summarise --scores $OUT/scores --pool N,R,S1,S2,S3,RS1,RS2,RS3`.
7. `report`: `compare_report.py` → `$OUT/compare.json`, `$OUT/compare.md`.
8. Copy the three `atlas_randdir_512_0.pt` of the shuffled runs to `$OUT/atlases/RS<c>.pt` and
   record their digests in `$OUT/atlases/digests.json`.

`SMOKE=1`: CPU, fp32, torch backend, ESM-2 8M from the local Hugging Face cache, `--max-seqs 32
--tokens-per-batch 4096 --pilot-tokens 8192 --K 16 --Kr 16 --edge-q 0.3 0.5 0.7 0.9 0.95 0.99`,
`PER_LAYER=4`, a synthetic whole-build descriptor file as `jobs/calib_8m/run.sh` builds one. The
job first builds its own stand-in for `CALIB` under `$OUT/smoke_inputs`: a trained run with
`--columns neurons_post,randdir:512:0` (giving N and R), and for each c a shuffled run with
`--columns neurons_pre,neurons_post` (the calibration's configuration), sampled and scored as
the calibration job does, into `smoke_inputs/pooling/`. `repro` is then a real check of
multi-source invariance.

## Acceptance (blind tests; invariants and arithmetic)

- `h1_verdict` gives the worked example's five outcomes, and at the boundaries: `d_NR = 0.147`
  with `p_NR = 0.0099` passes steps 1–2; `p_NR = 0.01` gives "against"; `ci_NR[0]` equal to an
  upper bound gives "gap not attributable to training"; `d_NR = -0.147`, `p_NR = 0.001` gives
  "strongly against".
- `bootstrap_cliffs`: disjoint samples give `[1.0, 1.0]` or `[-1.0, -1.0]`; two calls with the
  same arguments return the same interval; `draws=1` returns the single draw's value twice.
- `compare_report.py` on hand-built score directories (eight sources, synthetic records and
  scores): every `d` and `p` equals `cliffs_delta`/`mann_whitney` on the same values;
  `calibration_passes` is the conjunction of its three conditions; `h1` equals `h1_verdict`
  applied to the json's own numbers (and is null when calibration fails); `share_residue_mode`
  equals `separation_gate.stat_summary`.
- `SMOKE=1` passes end to end; `compare.json` has the frozen keys; `repro` reports every `N` and
  `S<c>` record identical (multi-source invariance); the test recomputes `h1` from the json.

## Cost

One `gpu_1x_a100_sxm4` in us-east-1 on `esm-atlas-results`: three shuffled runs about 13 min
each (the calibration's), eight scorings about 5.5 min each, setup 3 min: about 1.5 h. Hard cap
2.5 h, at most $4.98. Launch only with the owner's yes.
