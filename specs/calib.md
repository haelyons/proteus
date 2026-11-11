# Spec: 8M calibration atlases, vectorised randdir generator, pooling switch

Status: frozen at the commit that adds this file. Follows the sealed rule
(`results/science_decision_rule.md`): its calibration step needs post-GELU atlases of the trained
and shuffled-weight 8M models. The random-direction atlas is built here but **not scored** (the
four-source comparison is not built yet).

## 1. Vectorised `randdir` generator (`scripts/synthetic_columns.py`)

The `randdir` shape draws `torch.randn(n, 64, generator=extra)` once per column, in a Python loop
over columns. Replace the loop with one draw of shape `[C, n, 64]` from the same generator (the
same values in the same order), one GELU, and a per-column weighted sum. Required: for the same
arguments, `synthetic_atlas.pt` is **bitwise equal** to the looped version. If bitwise equality
cannot be met because the reduction order differs, keep `.sum(-1)` over a contiguous last
dimension of 64 (the looped version's reduction) and report the measured maximum difference;
do not change the stored values' definition.

### 1b. Amendment (2026-11-11, before any run): parallel per-protein generation

A local profile (1,490 columns, one thread) found the `randdir` build dominated by
`torch.randn` (about 0.7 ms per token, serial per generator) and GELU (about 0.5 ms per token):
about 10 thread-hours for the 24M-token design set, which vectorising does not reduce. Every
protein's values depend only on its own generators (`nseed(p)`, `nseed(p) + 2**40`), so blocks of
different proteins can be computed in parallel without changing any value.

- New argument `--gen-workers W` (default 1: the current code path, unchanged). With W > 1 the
  per-protein column blocks (`column_block`) of both the pilot pass and the stream pass are
  computed by W worker processes, each with `torch.set_num_threads(1)`, and consumed strictly in
  protein order, with at most 2W blocks in flight. Applies to every shape.
- The `randdir` draw may be split into consecutive column chunks (`randn(c, n, 64)` in column
  order, same generator) to bound memory; the drawn values and their order are unchanged.
- Required: `synthetic_atlas.pt` bitwise equal for W = 1 and W > 1, and to the code at the spec
  commit (section 1's requirement).
- `jobs/ruler_checks/run.sh` passes `--gen-workers` equal to the shape's thread count.
- Acceptance addition to (1): the same comparison with `--gen-workers 3` for `randdir` and
  `gauss`, bitwise.

## 2. `POOLING` switch (`jobs/ruler_checks/run.sh`)

`POOLING=0` skips the pooling steps (sample, scores, summaries); `checks.json` then has
`"pooling": null` (`scripts/ruler_checks_report.py` accepts a missing `pooling/`). Default
`POOLING=1` changes nothing. Used to rerun the `randdir` and `sae:0.001` shapes alone.

## 3. `jobs/calib_8m/run.sh`

Env `JOB OUT FS DEADLINE`; optional `WORKERS` (default `nproc - 2`), `PER_LAYER` (default 250).
Tolerates the inbox runner's files in `OUT` (as `jobs/shard_scale/run.sh`); `SCRATCH` defaults to
`$HOME/scratch/$JOB` with its parent created. Non-smoke preflight as `jobs/followup_650m/run.sh`
(pip upgrade, import check, `nvidia-smi` record, 1 Hz logger, GPU tests).

Atlas config, identical to the recorded 8M atlases (`jobs/batch_20261105/run.sh`, `C8`):

```
--model 8M --data $FS/cache/swissprot_2026_03.pt --device cuda --dtype bf16 --backend triton
--tokens-per-batch 32768 --K 64 --Kr 256 --edge-q 0.5 0.9 0.99 0.999 0.9999 0.99999
--exclude-acc heldout_created_after_2021-11-17.tsv --pilot-tokens 262144 --seed-thresholds --ckpt-every 2000
```

Steps:

1. `trained`: the config plus `--columns neurons_pre,neurons_post,randdir:512:0` →
   `$SCRATCH/trained/`.
2. `shuf<s>` for s = 1, 2, 3: the config plus `--control shuffled --control-seed s --columns
   neurons_pre,neurons_post` → `$SCRATCH/shuf<s>/`.
3. Reproduction check: `compare_atlas.py atlas_neurons_pre.pt <recorded atlas>` for each of the
   four runs, against `$FS/runs/atlas-batch-20261105-0258/a8_pre/atlas.pt`,
   `$FS/runs/atlas-batch2-20261105-0348/a8_shuf1/atlas.pt`,
   `$FS/runs/atlas-batch-20261105-0258/a8_shuf2/atlas.pt`,
   `$FS/runs/atlas-batch2-20261105-0348/a8_shuf3/atlas.pt` → `$OUT/repro_<name>.json`. A
   mismatch is recorded and does not stop the job.
4. Calibration scoring: `compare_sources.py sample --atlas trained=<trained atlas_neurons_post.pt>
   shuf1=... shuf2=... shuf3=... --per-layer PER_LAYER --seed 0`; `score` each with `--desc
   $FS/descriptors_swissprot_2026_03.pt --workers WORKERS` into `$OUT/pooling/scores/`; then
   `scripts/separation_gate.py --checks $OUT --out $OUT/calibration.json` (with an empty
   `$OUT/planted/`).
5. `$OUT/calib.json`: `{"repro": {NAME: {"ok", "exact_all"}}, "calibration": {CONTROL: {"d", "p",
   "passes"}}, "calibration_passes": bool, "randdir": {"digests", "directions_sha256"}}` where
   `passes = d >= 0.147 and p < 0.01` (the sealed rule), `calibration_passes` = all three.
6. Copy to `$OUT`: each run's `run.json` and `batches.jsonl`; the `randdir` atlas's `run.json`
   source entry. Large `.pt` files stay in `$SCRATCH` except
   `atlas_neurons_post.pt` and `atlas_randdir_512_0.pt` of `trained`, which are copied to
   `$OUT/atlases/` for later use (8M: about 0.6 GB each).

`SMOKE=1`: CPU, fp32, torch backend, ESM-2 8M from the local Hugging Face cache, `--max-seqs 64
--tokens-per-batch 4096 --pilot-tokens 8192 --K 16 --Kr 8 --edge-q 0.3 0.5 0.7 0.9 0.99`,
`PER_LAYER=4`, a synthetic whole-build descriptor file built under `$OUT/smoke_inputs` for the
Swiss-Prot test set's proteins, and the reproduction step compares against a pre-change
single-source run made in the job (`--columns` absent) instead of the recorded atlases.

## Acceptance (blind tests; invariants and arithmetic)

- (1) on a small synthetic descriptor set: `--shape randdir` output after the change equals the
  code at the spec commit bitwise (or the reported difference is at most 1e-6 relative, and the
  test records which held).
- (2) `POOLING=0 SMOKE=1` on the checks job writes `checks.json` with `pooling` null and runs no
  pooling step (`timeline.json` has no `pooling_*` step).
- (3) `SMOKE=1` calib job passes end to end; `calib.json` has the frozen keys; `repro` is ok for
  every run (the multi-source `neurons_pre` atlas equals the single-source run); `calibration_passes`
  equals the conjunction of its three `passes`, each recomputed from `calibration.json`.
