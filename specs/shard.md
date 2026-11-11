# Spec: sharded atlas (units 1–3)

Status: frozen at the commit that adds this file. Implementers and test writers work from this
text and the existing interfaces only. Where the spec and the code disagree, stop and report.

## Goal

Run one global batch plan on N processes (one per GPU), each taking whole batches, and merge the
N states into the atlas a single process would have produced. "Equal" means:

- **Exact fields, bitwise:** `top_val, top_tok, res_tok, res_val, res_key, sub_bin, enr,
  ann_tot, edges`, and the scalars `n_seen, n_sub, gram_rows`.
- **Float sums, within a relative tolerance:** `sub_sum, sub_sq, gram` (summation order differs).
- **Not compared:** `stats` counters (each shard filters against its own running thresholds,
  so candidate counts legitimately differ) and timings.

Preconditions for bitwise equality, which the tests must arrange and the code must not weaken:
every process sees the same global plan, the same bin edges, the same seeds computed from the
global stream size, and each batch has the same content and shape as in the single run.

Non-goals: threshold sharing between shards, multi-node launch, new column sources.

## Shared definitions

- **Global plan**: the list returned by `SwissProt.plan(...)` in `run_atlas.py` after
  `--only-acc`, `--exclude-acc`, `--max-seqs` and `--max-batches` are applied. Batch index `b`
  always means an index into this list.
- **Shard `i/N`**: `0 <= i < N`, `N >= 1`. It owns the batches with `b % N == i` and processes
  them in ascending `b`. Its position counter is `p = b // N`.
- **Plan digest**: `atlas.data.plan_digest(plan) -> str`, the sha256 hex digest of the bytes of a
  little-endian int64 array `[len(plan), len(g0), *g0, len(g1), *g1, ...]`.
- **Result digests**: `atlas.core.result_digests(res) -> dict[str, str]`, sha256 hex of
  `res[k].contiguous().numpy().tobytes()` for each `k` in `top_val, top_tok, res_tok, res_val,
  res_key, sub_bin, enr, ann_tot, edges` that is present. `run_atlas.py` must use it for the
  `digests` it already writes, with unchanged values.

## Unit 1: `--shard i/N` in `scripts/run_atlas.py` (plus small additions to `atlas/`)

### `atlas/core.py`

- `Atlas.state_dict(include_gram: bool = False)`. With `include_gram=True` it first calls
  `flush_gram()` and adds `gram64` (the fp64 host accumulator exactly as held, not symmetrised;
  absent when the config has no Gram) and `gram_rows`. Default output is unchanged.
- `Atlas.load_state_dict(st)`: when `st` has `gram64`, restore `gram64` and `gram_rows` from it
  instead of zeroing them. Without it, behaviour is unchanged.
- `Atlas.from_state(st: dict, device="cpu") -> Atlas` (classmethod). Builds an Atlas of the
  right shape from a state dict (`E`, `cfg`, `n_a`) and loads it. It must work on a host without
  CUDA or Triton whatever `cfg["backend"]` says, and `result()["cfg"]` must still report the
  original config values. Two atlases built by `from_state` from states with equal `cfg` must
  pass the config assertion in `merge_from`.
- `result_digests(res)` as defined above.

### `atlas/data.py`

- `plan_digest(plan)` as defined above.

### `scripts/run_atlas.py`: new arguments

| Argument | Meaning |
| --- | --- |
| `--shard i/N` | Process only the batches of shard `i/N`. Default: unset (all batches, behaviour unchanged). Malformed or out-of-range values are an argparse error. |
| `--pilot-out PATH` | After the pilot, write the pilot artifact to `PATH` (atomically). |
| `--pilot-only` | With `--pilot-out`: exit 0 after writing the artifact; no stream, no `atlas.pt`. |
| `--pilot-from PATH` | Load edges and seeds from a pilot artifact; run no pilot forward. Mutually exclusive with `--edges-from` and `--pilot-out`. |

Pilot artifact (a `torch.save` dict):

```
edges         float32 [L, 2, D, B]
top_seed      float32 [L, 2, D]      or None when the run has no --seed-thresholds
res_seed_key  int64   [L, 2, D, B]   or None likewise
n_total       int      residues in the global plan (as run_atlas computes n_total)
n_batches     int      len(global plan)
plan_sha256   str      plan_digest(global plan)
args_key      dict     the values of: model, control, control_seed, dtype, tokens_per_batch, window,
                       sorted_batches, pilot_tokens, K, Kr, seed, max_seqs, max_batches, edge_q,
                       seed_thresholds
model         dict     the `info` dict from atlas.esm.load
```

Rules:

1. `--pilot-from`: the process rebuilds its own global plan and exits non-zero with a message
   naming the first differing item if `plan_sha256`, `n_total`, or any `args_key` entry differs
   from its own. Seeds are applied exactly when the artifact holds them; if `--seed-thresholds`
   disagrees with the artifact, that is an `args_key` mismatch.
2. `--shard i/N` with `N > 1` requires `--pilot-from` (argparse error otherwise). Reason: a
   pilot repeated per GPU costs N pilots and makes equality depend on N bitwise-equal forwards.
3. The pilot, `n_total` and the seeds always refer to the **global** plan, never the shard's
   share.
4. `--max-batches` truncates the global plan before sharding. `--plan-offset b0` starts at the
   first owned batch `>= b0`. `--stop-after M` counts batches this process handled.
5. Gram flush: without `--shard` the existing rule holds (`b % gram_flush_every ==
   gram_flush_every - 1`). With `--shard` it is `p % gram_flush_every == gram_flush_every - 1`.
   For `0/1` the two coincide.
6. Checkpoints and `--resume` work under `--shard`. `next_batch` stays a global index: resume
   continues with the first owned batch `>= next_batch`. `--shard` is part of the resume
   `args_key` when set (unset runs keep their old key, so old checkpoints still resume).
7. Output. Without `--shard`: unchanged (`atlas.pt`, `run.json`, `batches.jsonl`). With
   `--shard` (including `0/1`): at completion write `shard.pt` atomically and no `atlas.pt`:

   ```
   atlas          Atlas.state_dict(include_gram=True)
   shard          [i, N]
   plan_sha256, n_batches_plan, n_total_plan
   batches_done   number of batches this shard merged into its state
   args_key       the resume args_key
   model, coverage, vocab, order_seed      as run_atlas puts into atlas.pt
   ```

   `run.json` additionally carries `shard: [i, N]`, `plan_sha256` and `batches_done`.
8. A shard that owns no batch (N larger than the plan) completes and writes a valid, empty
   `shard.pt`.

### `scripts/run_atlas.py`: instrumentation (all runs, sharded or not)

- Each `batches.jsonl` record gains `shard` (`[i, N]` or null), `shard_pos` (`p`, or `b` when
  unsharded), `build_s` (host seconds to build the batch and its host-side context, measured
  where they are built, also on the prefetch thread), `wait_s` (seconds the main loop waited for
  that batch), and on CUDA `gpu_mem_alloc` and `gpu_mem_reserved` (bytes, current).
- `run.json` gains `seconds.final_save`, `seconds.total` (process start to finish) and, on CUDA,
  `gpu` = `{name, total_memory, max_memory_allocated, max_memory_reserved, uuid if available}`;
  `host` = `{cpus, peak_rss_bytes}`.
- Existing record fields keep their names and meaning. `batch` stays the global index.

### Acceptance (unit 1)

A. Without the new flags, a run's `atlas.pt` exact fields, float sums and `digests` are bitwise
   what they were before the change (same args, CPU, fp32, one thread).
B. `--pilot-out` then `--pilot-from` (unsharded) gives an `atlas.pt` bitwise equal in all
   compared fields, float sums included, to a plain run with the same args.
C. `--pilot-from` with a different `--seed`, `--K`, `--max-seqs` or `--tokens-per-batch` exits
   non-zero before any streaming.
D. For N in {2, 3}: the union of `batch` values in the N shards' `batches.jsonl` is exactly
   `0..len(plan)-1`, pairwise disjoint, each shard's ascending, each with `b % N == i`.
E. A shard stopped with `--stop-after` and resumed gives a `shard.pt` whose `atlas` state equals
   the uninterrupted shard's in every `_STATE` field and in `enr, ann_tot, n_seen, n_sub`.

## Unit 2: `scripts/merge_atlas.py`

```
python scripts/merge_atlas.py --out DIR SHARD_PT [SHARD_PT ...] [--device cpu] [--allow-partial]
```

- Loads the `shard.pt` files, sorts them by `i`, and folds them left to right with
  `Atlas.merge_from`: `acc = from_state(s0); acc.merge_from(from_state(s1)); ...`. Each step is a
  pairwise merge; at most two states are alive at a time (the folded-in state is released before
  the next load). Files are loaded one at a time, not all up front.
- Refuses (non-zero exit, message, nothing written) when: `N` differs between files; an index
  is duplicated; indices do not cover `0..N-1` (unless `--allow-partial`); `plan_sha256`,
  `n_batches_plan`, `n_total_plan` or `args_key` (ignoring its `shard` entry) differ; or the sum
  of `batches_done` is not `n_batches_plan` (unless `--allow-partial`). Differing edges or seeds
  are caught by `merge_from`'s own assertions; the script must not bypass them.
- Never writes into or modifies the input files or their directories unless `--out` is one of
  them (not recommended, not forbidden).
- Writes to `DIR`:
  - `atlas.pt`: `Atlas.result()` of the merged state plus `model, coverage, vocab, order_seed,
    n_total_plan, n_batches`, i.e. the same keys a single run writes. Atomic write.
  - `merge.json`: `shards` (per input: path, `[i, N]`, `batches_done`, `n_seen`, `stats`, file
    bytes), `steps` (per fold step: `load_s`, `merge_s`, `rss_after_bytes`), `result_s`,
    `save_s`, `total_s`, `peak_rss_bytes` (process high-water mark, `resource.getrusage`),
    `digests` (`result_digests`), `n_seen`, `n_sub`, `gram_rows`, `stats` (summed),
    `res_seeded_bins`, `res_seed_limited_bins`, `top_seed_short_groups`, `atlas_pt_bytes`.
- One shard `0/1` is a valid input; the output equals that shard's own result.

### Acceptance (unit 2)

F. For N in {1, 2, 4}, ESM-2 8M, CPU, fp32, one thread, seeded thresholds, shared pilot
   artifact: merged `atlas.pt` equals the single-run `atlas.pt` bitwise in every exact field and
   scalar, and `sub_sum, sub_sq, gram` agree to relative 1e-9 or better (fp64 sums of identical
   per-batch terms).
G. Merging the same shards in a different command-line order gives a bitwise identical
   `atlas.pt` in all compared fields, float sums included (the script sorts by `i`).
H. Each refusal case above exits non-zero and leaves `DIR` without `atlas.pt`.

## Unit 3: the gate

### `scripts/compare_atlas.py`

Existing positional use keeps working, with the same exit codes and the same JSON keys. New:

- `--rtol X` (default `1e-5`): threshold for the float-sum fields. Each such field's report
  gains `ok`.
- `--digests RUN_JSON`: instead of a second atlas, compare `result_digests` of the first file
  against `RUN_JSON["digests"]`. Reports per field `{exact, a, b}`; fields missing on either
  side are listed under `missing` and make the gate fail. Scalars `n_seen` and `gram_rows` are
  compared when `RUN_JSON` has them. Float sums cannot be checked this way; the report says so
  (`close_checked: false`).
- The report gains top-level `exact_all`, `close_all`, `scalars_all`, `rtol`. `ok` is their
  conjunction (in `--digests` mode: `exact_all and scalars_all`).
- A field present in exactly one of two files is reported under `missing` and fails the gate
  (today it is silently skipped). Exception: `top_seed, res_seed, top_s, top_t, top_k, res_s,
  res_t, res_v, E` may be absent on both sides (result-form files).
- For exact float fields NaN compares equal to NaN, as today.

### `scripts/batch_digest.py`

```
python scripts/batch_digest.py --model M --data D --batch B [B ...] --out J [--device ...] [--dtype ...]
    [run_atlas plan arguments: --seed --tokens-per-batch --window --max-seqs --exclude-acc --max-batches]
```

For each requested global batch index it runs one forward with the atlas hook point attached and
writes, per batch and layer, the sha256 of the raw bytes of the hooked `[T, D]` tensor in its
native dtype, plus `plan_sha256`, batch shape, device name and UUID, torch and CUDA versions.
`python scripts/batch_digest.py --compare J1 J2` exits 0 when every digest matches and prints
the first differing (batch, layer) otherwise. This is the two-GPU bitwise check.

### `jobs/shard_gate/run.sh`

Env `JOB OUT FS DEADLINE`, optional `NSHARDS` (default 2), `SMOKE` as in the other jobs.

1. Pilot once (`--pilot-out`, `--pilot-only`).
2. Single run from the artifact (the reference).
3. `NSHARDS` shard runs from the same artifact. On GPU: shard `i` on device `i`, concurrently.
   On `SMOKE=1`: ESM-2 8M, CPU, fp32, one thread, sequentially.
4. `merge_atlas.py`.
5. `compare_atlas.py merged single --out gate_compare.json`; on GPU also `batch_digest.py` for
   one batch on device 0 and device 1, and `--compare`.
6. Exit non-zero if the gate fails. Writes `timeline.json` with start/end of each step.

It must refuse to write outside `$OUT`, and it reads the dataset and model caches only.

### Acceptance (unit 3)

I. `compare_atlas.py` exits 0 on a file against itself and against its own run's `run.json`
   digests; exits 1 when one element of `top_tok` is changed, when `sub_sum` is perturbed by a
   relative 1e-3, and when a digest in the JSON is altered.
J. `--rtol 1e-2` accepts a relative 1e-3 perturbation of `sub_sum` that the default rejects.
K. `batch_digest.py` run twice on CPU (one thread) gives matching files; `--compare` exits
   non-zero when one digest is edited.
L. `SMOKE=1 bash jobs/shard_gate/run.sh` passes end to end on a 500-sequence dataset.

## Test conventions

- Tests live in `tests/test_shard_*.py`, runnable both as `python -m pytest tests/test_shard_x.py`
  and `python -m tests.test_shard_x` (exit code 0/1), like the existing tests.
- Real-model tests use ESM-2 8M on CPU with `--dtype fp32`, `OMP_NUM_THREADS=1` and
  `MKL_NUM_THREADS=1`, and the dataset at `$PROTEUS_TEST_DATA` (default
  `data/swissprot_test500.pt`; skip with a clear message when absent). Keep each test file under
  about ten minutes on four cores: small `--max-seqs`, `--tokens-per-batch 4096` or `8192`,
  `--K 16 --Kr 8`.
- Tests assert on files and exit codes. No test may pass by construction (e.g. comparing a file
  with itself where the spec asks for two runs).
