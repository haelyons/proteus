# Spec: scaling benchmark (unit 4)

Status: frozen at the commit that adds this file. Builds on `specs/shard.md` (units 1–3, done).

## Goal

On one host with G identical GPUs, measure the sharded atlas at N = 1, 2, 4, 8 against the
single-process run of the same workload, gate every merged atlas against that run, and separately
measure filtered against `--naive` on one GPU. Every reported number comes from a timer, a
counter or a file comparison written by the job. Nothing is launched by this unit; the job must
pass `SMOKE=1` locally first.

## `jobs/shard_scale/run.sh`

Env: `JOB OUT FS DEADLINE` as in the other jobs. Optional:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NS` | `"1 2 4 8"` | shard counts; values above the visible GPU count are skipped and logged |
| `SCRATCH` | `$HOME/scratch/$JOB` | local disk for `shard.pt`, `atlas.pt`, `pilot.pt`; never on `$FS` |
| `NAIVE_PREFIX` / `NAIVE_STEADY` / `NAIVE_OFFSET` | `60` / `40` / `4500` | batches for the filtered-vs-naive pairs |
| `DIGEST_BATCHES` | `"0 1000"` | global batch indices for the cross-GPU bitwise check |
| `REF_RUN_JSON` | `ref_run.json` in the job directory | committed `run.json` with field digests of the recorded reference atlas |
| `KEEP` | `0` | `1`: copy the N=1 `atlas.pt` and the largest-N merged `atlas.pt` to `$OUT` at the end |
| `SMOKE` | `0` | `1`: ESM-2 8M, CPU, fp32, one thread, `NS="1 2"`, shards sequential, a few batches, no GPU steps |

Frozen workload (non-smoke), identical to the recorded 650M reference
(`results/followup_650m_20261104/full/run.json`): `--model 650M --dtype bf16 --backend triton
--tokens-per-batch 32768 --pilot-tokens 262144 --K 64 --Kr 64 --edge-q 0.5 0.9 0.99 0.999 0.9999
0.99999 --seed-thresholds --seed 0 --exclude-acc heldout_created_after_2021-11-17.tsv --prefetch 2
--stats-impl v3 --merge-impl stacked --gram-flush-every 1000`, data
`$FS/cache/swissprot_2026_03.pt`. All runs use `--ckpt-every 1000000` (no checkpoints: the cap
is short and checkpoints would be a cost only the N=1 run pays). Model and workload values are
overridable by env (`MODEL TPB PT K KR`) so the same job can run 3B later.

Rules:

- Writes only under `$OUT` (small records) and `$SCRATCH` (large files). Reads `$FS/cache` only.
  Never deletes or overwrites anything else on `$FS`. Refuses a non-empty `$OUT`.
- Online preflight on GPU as in `jobs/followup_650m/run.sh` (pip upgrade of pillow and
  transformers, import check). `HF_HUB_OFFLINE` only under `SMOKE=1`. Also fix the same two
  points in `jobs/shard_gate/run.sh` (its GPU path currently forces offline pip and offline HF),
  and make its GPU limits (`--max-seqs`, `--max-batches`, K, Kr, tokens, pilot) env-overridable.
- Every step goes through a `step` wrapper that records start, end and return code into
  `timeline.json` (as `jobs/shard_gate/run.sh` does), honours `DEADLINE`, and logs to
  `$OUT/<step>.log`. A failed gate or shard does not stop later N values; the job's exit code is
  non-zero if any step failed.

Steps:

1. **Host and GPU record** (`$OUT/host/`): `nvidia-smi --query-gpu=index,uuid,serial,name,
   pci.bus_id,driver_version,vbios_version,memory.total,power.limit,power.default_limit,
   clocks.max.sm,ecc.mode.current,compute_mode,temperature.gpu --format=csv` to `gpus.csv`;
   full `nvidia-smi -q` to `nvidia_smi_q.txt`; `lscpu`, `numactl -H` (if present), `free -b`,
   `df` of `$SCRATCH` and `$FS`, mount options of `$FS`, kernel, and torch / CUDA / cuDNN /
   triton / transformers versions to `host.json`. A 1 Hz logger for all GPUs
   (`timestamp,index,uuid,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,
   utilization.memory,memory.used`) to `$OUT/smi.csv` for the whole job, liveness checked as in
   `jobs/followup_650m/run.sh`.
2. **GPU tests** on device 0 (the two existing `--cuda` test modules).
3. **Cross-GPU bitwise check**: `batch_digest.py` for `DIGEST_BATCHES` on every visible GPU
   concurrently; `--compare` each against device 0; summary to `$OUT/digest_compare.json`
   (`{"all_equal": bool, "pairs": [{"gpu": i, "equal": bool, "first_diff": ...}]}`). A mismatch
   is recorded and does not stop the job.
4. **Pilot** once on device 0: `--pilot-out $SCRATCH/pilot.pt --pilot-only`.
5. **N = 1**: plain unsharded run from the pilot artifact on device 0 → `$SCRATCH/single/`.
   Then `compare_atlas.py single/atlas.pt --digests $REF_RUN_JSON` → `$OUT/single_vs_ref.json`.
6. **For each N > 1 in `NS`**: N shard processes concurrently, shard `i` on device `i`, all
   from the same artifact → `$SCRATCH/n<N>/s<i>/`; wall clock from first launch to last exit is
   its own timeline entry (`shards_n<N>`). Then `merge_atlas.py` under `/usr/bin/time -v` →
   `$SCRATCH/n<N>/merged/`; `compare_atlas.py merged single --rtol 1e-5` →
   `$OUT/n<N>/gate_compare.json`; `compare_atlas.py merged --digests $REF_RUN_JSON` →
   `$OUT/n<N>/vs_ref.json`. Copy each run's `run.json`, `batches.jsonl`, `merge.json` and logs to
   `$OUT/n<N>/...`; delete that N's `shard.pt` files from `$SCRATCH` after its compare.
7. **Filtered vs naive**, one GPU each, run concurrently on devices 0 and 1 (devices 0 and 0
   sequentially when only one GPU): prefix pair (`--max-batches $NAIVE_PREFIX`, own pilot,
   seeded filtered vs `--naive`), `compare_atlas.py` of the two → `$OUT/naive/cmp_prefix.json`;
   steady pair (`--plan-offset $NAIVE_OFFSET --stop-after $NAIVE_STEADY`, naive with fresh state;
   filtered from the pilot artifact with fresh state) for timing only.
8. `scripts/scaling_report.py $OUT`.

## `scripts/scaling_report.py OUT [--predictions P.json]`

Reads only files under `OUT`; writes `OUT/scaling.json` and `OUT/scaling.md`. Missing inputs
give `null` entries, not a crash. Definitions:

- `pilot_s`: the pilot step's wall time from `timeline.json`.
- `wall_1 = pilot_s + (N=1 run step wall)`. `wall_N = pilot_s + shards_n<N> wall + merge step
  wall`. Compare steps are reported but excluded from both.
- `speedup_e2e = wall_1 / wall_N`; `efficiency = speedup_e2e / N`.
- `speedup_stream = stream_1 / max_i stream_i` with `stream = run.json seconds.stream_this_process`.
- Per run (single and every shard): `seconds.{load, pilot, stream_this_process, final_save,
  total}`, sums over `batches.jsonl` of `atlas_pass_s, build_s, wait_s, seconds`, batch count,
  redo layer count, `stats.compacted / top_cand / res_cand`, `gpu.max_memory_allocated`,
  `gpu.max_memory_reserved`, `gpu.uuid`, `host.peak_rss_bytes`, output file bytes; from `smi.csv`
  over the run's stream window and its GPU: mean and p5 utilisation, mean power, max memory used.
- Per N: `candidates_ratio = sum_i compacted_i / compacted_single` (same for `top_cand`,
  `res_cand`); imbalance `max_i stream_i / mean_i stream_i`; merge: `validate_s`, per-step
  `load_s` and `merge_s`, `result_s`, `save_s`, `total_s`, `peak_rss_bytes`, and `time -v`
  maximum resident set size; gate: `ok`, `exact_all`, `close_all`, per-field `max_rel_diff`;
  `vs_ref.ok`.
- Naive: per pair, `sum(seconds)` and `sum(atlas_pass_s)` of naive over filtered on the same
  batches; `cmp_prefix.ok`.
- With `--predictions`: each predicted quantity next to its measured value and their ratio. The
  script computes no verdicts.

`scaling.md`: two separate tables, "N versus 1" and "filtered versus naive", plus the gate and
cross-GPU results, each cell traceable to a key in `scaling.json`.

## `scripts/predict_scaling.py`

```
python scripts/predict_scaling.py --batches B.jsonl --run RUN.json --ns 1 2 4 8 \
    --fixed '{"load_s":..,"shard_save_s":..,"merge_load_s":..,"merge_step_s":..,"result_s":..,"save_s":..}' --out P.json
```

Pure arithmetic on a recorded single-GPU run, no model: for each N and shard i, the sum of
`atlas_pass_s` over batches with `b % N == i`; `stream_pred = max_i`; `speedup_stream_pred`;
`wall_pred = pilot + load + stream_pred + shard_save + N * (merge_load + merge_step) + result +
save` (N = 1: `pilot + load + stream + save`); `speedup_e2e_pred`; and the predicted candidate
ratio per N, by evaluating the recorded `compacted_frac(n_seen)` curve (25-batch moving average,
linear interpolation) at each shard's own `n_seen`. Output keys must match the names
`scaling_report.py --predictions` reads.

## Acceptance

M. `SMOKE=1 bash jobs/shard_scale/run.sh` passes on the 500-sequence dataset in under about
   12 minutes on four cores and leaves `scaling.json` with non-null `speedup_e2e`,
   `speedup_stream`, `candidates_ratio`, merge `peak_rss_bytes`, per-run `build_s`/`wait_s` sums
   for N = 1 and 2, a passing `n2/gate_compare.json`, and a naive section with `cmp_prefix.ok`.
N. `scaling_report.py` on a hand-built `OUT` tree with known numbers reproduces `wall_N`,
   `speedup_e2e`, `speedup_stream`, `candidates_ratio` and imbalance exactly, and survives a
   missing `n4/` directory.
O. `predict_scaling.py` on a synthetic 16-batch `batches.jsonl` gives the hand-computed
   per-shard sums, `stream_pred` and `wall_pred` for N = 1, 2, 4.
P. `SMOKE=1` never writes outside `$OUT` and `$SCRATCH`, and the job refuses a non-empty `$OUT`.

## Addendum: layout and step names (fixed after the blind tests were written; implementers follow it)

- `$OUT/single/{run.json,batches.jsonl}`, `$OUT/n<N>/s<i>/{run.json,batches.jsonl}`,
  `$OUT/n<N>/merged/merge.json`, `$OUT/n<N>/gate_compare.json`, `$OUT/n<N>/vs_ref.json`,
  `$OUT/single_vs_ref.json`, `$OUT/naive/...`, `$OUT/host/...`.
- `timeline.json` has the envelope of `jobs/shard_gate/run.sh` (`{"job": ..., "steps": [{"step",
  "start", "end", "rc"}]}`), with steps named `pilot`, `single`, `shards_n<N>`, `merge_n<N>`,
  `compare_n<N>`, `vs_ref_n<N>` (others free). Step times may be fractional seconds.
- `scaling.json` holds one record per N under a key that contains N (e.g. `"n": {"2": {...}}`),
  with the metric names of this spec as keys; N values without inputs appear with `null` metrics.
- Merge peak RSS applies to N > 1 only (N = 1 is the unsharded run, no merge).
