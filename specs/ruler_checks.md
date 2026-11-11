# Spec: ruler checks job (5a measurement and unit 6 checks)

Status: frozen at the commit that adds this file. Builds on `specs/planted.md` and
`specs/compare.md`. `atlas/ruler.py` stays byte-identical.

## Inputs (read only; recorded paths on `esm-atlas-results`, us-east-1)

| Input | Path under `$FS` | Recorded in |
| --- | --- | --- |
| Swiss-Prot cache | `cache/swissprot_2026_03.pt` | `INVENTORY.md` |
| Descriptor subset (ruler design proteins) | `runs/atlas-wab-20261105-0140/desc_subset/` | `results/README.md`, measurement #2 |
| Whole-build descriptors | `descriptors_swissprot_2026_03.pt` | `autolabel/esm/box_run.sh` |
| 8M trained atlas | `runs/atlas-batch-20261105-0258/a8_pre/atlas.pt` | same |
| 8M shuffled-weight atlases | `runs/atlas-batch-20261105-0258/a8_shuf{1,2,3}/atlas.pt` | `jobs/batch_20261105/run.sh` |
| UniRef50 map | `lists/swissprot_2026_03_uniref50.tsv.gz` (in the job) | repo |

The job never writes under `$FS` outside `$OUT` and never deletes there.

## Addition: `scripts/compare_sources.py calls`

```
python scripts/compare_sources.py calls --scores SCORES_DIR --source NAME [--rule v1.2] --out CALLS_JSON
```

Loads `SCORES_DIR/NAME.scores.pt` and `NAME.jsonl`, runs `calls(scores, rule)` once over that
source's scores in sample order, and writes `{"rule": RULE, "calls": [...], "residues": T_n_or_null,
"seconds": ...}` in `scripts/run_ruler.py`'s format: each call record has `col` set to the
column's unit index (for a one-layer synthetic atlas this is the synthetic column index) and the
same `levels` block `run_ruler.py` adds. For a one-layer atlas sampled in full and scored with
the same Tables and B, its `calls` list equals `run_ruler.py`'s for that atlas. This avoids
scoring each synthetic atlas twice.

## `jobs/ruler_checks/run.sh`

Env `JOB OUT FS DEADLINE`; `SMOKE=1` runs every step on a small synthetic descriptor set, a tiny
Swiss-Prot dict and tiny atlases built by the job itself under `$OUT/smoke_inputs`, in under 10
minutes on four cores. Optional `WORKERS` (default `nproc - 2`), `SHAPES` (default
`gauss gelu gelu_off randdir sae:0.1 sae:0.01 sae:0.001`), `PER_LAYER` (default 250).

Steps (each through the `step` wrapper of `jobs/shard_scale/run.sh`, `timeline.json` as there):

0. `inputs`: for every input above, path, size and sha256 into `$OUT/inputs.json`; a missing
   input fails the job here.
1. Planted shapes, per shape (shapes run concurrently, `WORKERS / number of shapes` threads each):
   `synthetic_columns.py --data DATA --desc-dir DESC_SUBSET --uniref UNIREF --shape S
   [--sae-rate R] --mixtures --noise-columns 500 --out $OUT/planted/<S>` (other arguments at
   their defaults, as measurement #2); then `compare_sources.py sample --atlas
   <S>=<that synthetic_atlas.pt> --per-layer C --seed 0`; `score` with `--desc DESC_SUBSET`;
   `calls --rule v1.2`; `eval_synthetic.py --calls ... --columns ... --out gate.json`;
   `summarise --pool <S>`.
2. Pooling check, 8M: `sample --atlas trained=A8_PRE shuf1=A8_SHUF1 shuf2=A8_SHUF2
   shuf3=A8_SHUF3 --per-layer PER_LAYER --seed 0`; `score` each source with `--desc` the
   whole-build descriptor file; `trained` is scored twice, with `WORKERS` and with 4 workers, into
   separate directories; then `summarise` three times: `--pool trained`, `--pool trained,shuf1`,
   `--pool trained,shuf1,shuf2,shuf3`.
3. `scripts/ruler_checks_report.py $OUT` writes `$OUT/checks.json` and `$OUT/checks.md`:
   - per shape: the v1.2 gate table (`gate.json`), per type x alpha the v1.2 categories and the
     share with `p_struct_corr <= 0.01`, for mixtures and parts the median `primary_jsd` and the
     residue modes, and the noise null rate `k/n` of `p_struct_corr <= 0.01` with its
     Clopper-Pearson interval;
   - pooling: `records_identical` (the two `trained` scorings, every record field), the trained
     source's v1.2 call counts in each of the three pools, and per pool the trained source's
     `share_struct` (must not change: it does not depend on the pool).

### `checks.json` (frozen keys)

```
{"shapes": {SHAPE: {"gate": {...as gate.json "gate"...},
                    "by_type_alpha": {"TYPE@ALPHA": {"calls": {CALL: n}, "n": n, "k_struct": k, "jsd_q50": x,
                                                     "residue_modes": {RES: n}}},
                    "noise_null": {"k": k, "n": n, "rate": r, "ci": [lo, hi]}}},
 "pooling": {"records_identical": bool, "n_trained": n,
             "trained_calls": {"trained": {CALL: n}, "trained,shuf1": {...}, "trained,shuf1,shuf2,shuf3": {...}},
             "trained_share_struct": {POOL: x}},
 "inputs_sha256": {NAME: hex}}
```

Worked example. A shape with 500 extra noise columns plus the 18 noise columns of the base types,
of which 4 have `p_struct_corr <= 0.01`: `noise_null = {"k": 4, "n": 518, "rate": 0.00772,
"ci": [0.00211, 0.01965]}` (Clopper-Pearson 95%). Columns of type `noise` at every alpha
count, alpha being irrelevant to them.

## Cost

`gpu_1x_a10` (30 vCPU, 200 GB RAM) in us-east-1 with `esm-atlas-results` attached; the GPU is
unused. Estimate 1.5–2.5 h; hard cap 4 h, at most $5.16. Launch only with the owner's approval.

## Tests (blind; invariants and arithmetic)

- `calls` adapter: on a small synthetic one-layer atlas and synthetic Tables, its `calls` list
  equals `run_ruler.py --rule v1.2`'s.
- `ruler_checks_report.py` on hand-built inputs reproduces `noise_null` (worked example),
  `k_struct` per type and alpha, and `records_identical` both true and false (one field changed).
- `SMOKE=1` passes end to end and writes only under `$OUT`.
