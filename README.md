# proteus

Per-neuron atlas of ESM-2 FFN units over Swiss-Prot, computed in one streaming pass without
materialising activations. Claim under test: an exact, monotone-threshold filter turns a ~70 TB
problem (168,960 FFN units x ~2e8 residues in bf16, for ESM-2 650M) into a one-GPU job of about an
hour. Measured (ESM-2 650M, Swiss-Prot 2026_03 minus a held-out set: 194M tokens, 168,960 units x 2
poles): one A100 40 GB, 60.4 min of forward plus atlas (forward alone 48.8 min), 71 min end to end with
pilot, checkpoints and save. Every measurement, next to its prediction, is in
[`results/README.md`](results/README.md).

## What is computed

Definitions (each quantity is exact with respect to its definition, for the activations the model
actually computed):

| quantity | definition | how |
|---|---|---|
| top-K | per (layer, pole, unit) the K largest s = +h / -h over valid tokens, ties broken by the smaller token id | an int64 key (s, then -token id) orders tokens strictly; the running K-th key only rises, so `s >= tau` admits every token that can still enter, in any arrival order |
| reservoir | per (layer, pole, unit, bin) the K_r tokens with the smallest key u(token) among tokens with s in the bin | bottom-k on a bijective 32-bit hash of the packed token id: a uniform, reproducible sample with no key ties; same monotone argument |
| subsample statistics | sums, sums of squares, bin masses, annotation contingency counts, Gram | exact on the hashed subsample u < rho_sub * 2^32 (estimates of full-data quantities; the Gram is structure, not statistics) |

Hook point: `encoder.layer[i].intermediate.dense` output (pre-GELU). Both poles are kept; GELU
barely transmits the negative pole (min ~ -0.17), so it is reported as secondary. Bins: lower
edges per (layer, pole, unit) at pilot quantiles (0.5, 0.9, 0.99, 0.999, 0.9999) of s, from a
pilot of randomly chosen sequences (all lengths). Tokens: id = seq_idx * 1024 + residue position;
sequences longer than 1022 residues are truncated and coverage is reported (Swiss-Prot 2026_03:
3.21% of sequences, 5.15% of residues dropped).

### Two paths per batch and layer: filtered and dense

- **Filtered (normal) path.** The forward hook does one compare-and-compact pass over the layer's
  activations H (fused Triton kernel, or `torch.nonzero`), keeping only elements above
  min(top-K tau, the lowest edge of any reservoir bin whose key ceiling is still above the
  bulk-row cutoff). Reservoir bins that are already well filled are served from the hashed bulk
  rows (key < rho_res * 2^32) instead. Row-level decisions (valid rows, token ids, keys, subsamples)
  are made on the host before the forward; candidates from all layers are merged once after it.
- **Dense path.** If a layer's candidates exceed its buffer (0.5% of elements), that layer is
  re-run for this batch with exact per-column `topk` and per-bin bottom-k, then the stream goes back
  to the filtered path. Both paths give the same result; the dense one is slower (about 28x a
  forward pass for the whole model on an A10). It is taken during warm-up and on bursts: in
  practice early layers (0-1) on the first short-protein batches and on single compositionally
  extreme proteins (poly-Ser, Gly-rich), where early-layer activations shift far into the tail.

### Seeded thresholds (`--seed-thresholds`)

- **Top-K** starts at the 2K-th largest pilot value of each unit. The pilot tokens are streamed
  too, so the final K-th value can only be at or above it: exact by construction (the 2K margin also
  absorbs last-bit differences between the pilot forward and the streamed forward).
- **Reservoir** bins get a key ceiling set so that the expected number of keys under it over the
  whole stream is c * K_r (c = 4), from a lower bound on the bin's pilot mass that accounts for
  tail tokens clustering within proteins (n_eff = (sum c_i)^2 / sum c_i^2 over pilot proteins).
  Bottom-k is order-independent, so a bin that ends with at least K_r keys under its ceiling holds
  exactly the unseeded result; `verify_seeds()` reports every bin that ends short (it then holds all
  of the bin's keys under the ceiling, a uniform sample, not the bottom K_r).
- Effect on an A10 (ESM-2 650M): first-120-batch pooled overhead 2.43 -> 0.64; all 1,543,826 seeded
  bins verified.

### Checkpoint, resume, merge, determinism

- `scripts/run_atlas.py` writes atomic checkpoints every N batches (state, pilot edges, seeds, batch
  offset; about 1.9 GB for 650M) and stops cleanly before `--deadline`; `--resume` continues from the
  last checkpoint. The Gram is not checkpointed: it is written once at the end with its row count.
- `Atlas.merge_from` combines runs over disjoint data exactly (shards must share seeds computed from
  the global stream size).
- On GPU the forward pass is bitwise reproducible across repeats (bucketed and heavily padded
  batches). Multi-threaded CPU SDPA is not, on padded batches; CPU tests that compare two runs pin
  one thread.

### Statistics implementations

`AtlasConfig.stats_impl = "v3"` (default) runs two Triton kernels (`atlas/stats_triton.py`) on CUDA:
one pass over the subsample rows that keeps per-column bin counts and moments in registers (fp32
per 16-row block, fp64 across blocks) and writes the int8 enrichment masks in the layout of one exact
int8 GEMM; and a Gram kernel that accumulates the upper block triangle of X^T X in place (bf16 tensor
cores, fp32 accumulation; symmetrised once at the end). On CPU v3 falls back to `"v1"`, the reference,
unless `TRITON_INTERPRET=1`. `"v2"` (searchsorted bins, int64 index arithmetic) was measured no faster
than v1 on the A10 (340 vs 333 ms per batch: the int64 [D, n] passes cost as much as the booleans they
replaced) and is kept only for the record. `merge_impl = "stacked"` (default) builds reservoir
candidates from the bulk rows across all layers at once (-32 ms per batch on the A10); `"loop"` is the
per-layer reference. All are in the exactness tests, including tie-heavy data in shuffled arrival
order. `jobs/science_650m` checks v1 + loop against v3 + stacked on real data (checkpoints bitwise equal
except sums, which agree to < 1e-5) before the full pass.

### Held-out set

`lists/heldout_created_after_2021-11-17.tsv`: the 10,010 Swiss-Prot entries created after UniProt
release 2021_04 (17 Nov 2021), the UniRef50 release ESM-2 was trained on (2.06% of residues).
`run_atlas.py --exclude-acc` leaves them out of the stream, for held-out prediction tests.

## Layout

- `atlas/core.py` - Atlas: streaming state, both paths, seeds and verification, merges, checkpoint/merge
- `atlas/filter_triton.py` - fused compaction kernel (early exit on empty tiles; tuned tile 8x512)
- `atlas/stats_triton.py` - subsample statistics kernel and upper-triangle Gram kernel (stats v3)
- `atlas/reference.py` - offline ground truth, pilot edges, pilot seeds
- `atlas/data.py`, `atlas/esm.py` - Swiss-Prot batching plan and pilot plan; ESM-2 loading, controls, hooks
- `scripts/build_swissprot.py` - sequences + coarse residue annotations (UniProt REST)
- `scripts/run_atlas.py` - end-to-end run (`--seed-thresholds`, `--pair-forward`, `--profile-at`, `--resume`, `--deadline`)
- `scripts/profile_forward.py` - forward kernel breakdown, determinism, GEMM microbench, tile sweep
- `scripts/bench_phase1.py`, `scripts/diag_candidates.py`, `scripts/diag_redo.py`, `scripts/analyze_a10.py` - measurements behind `results/`
- `scripts/bench_gram.py` - subsample GEMMs (TF32 vs bf16, int8 enrichment) and the one-pass read floor at the per-layer shape
- `scripts/compare_atlas.py` - two checkpoints of the same stream: exact fields bitwise, sums by relative difference
- `atlas/proftrace.py` - per-range attribution from a profiler trace: span vs busy device time, top kernels per range
- `autolabel/` - first-read notebooks (InceptionV1 replay; ESM-2 8M/650M/3B from the atlases), the 3B atlas job, and notes on applying Cammarata's approach to automated labelling (`autolabel/README.md`)
- `tests/` - exactness (fp32/bf16, overflow redo, seeded, too-tight seeds, tie-heavy shuffled order),
  Triton kernels (compaction; statistics and Gram, interpreter or `--cuda`), merge, resume
- `jobs/` - GPU jobs; each `run.sh` supports `SMOKE=1` (same steps, ESM-2 8M, CPU): `a10_profile`
  (forward profile, long run), `impl_ab` (stats v1 vs v2, merge loop vs stacked, same seeds and plan),
  `science_650m` (GPU tests, equivalence gate on real data, full Swiss-Prot pass with the frozen config)
- `infra/` - Lambda Cloud over HTTPS only (below)
- `results/` - curated measurement records

## Running

CPU checks:

```
python -m tests.test_exactness
TRITON_INTERPRET=1 python -m tests.test_triton_kernel
python scripts/build_swissprot.py --out data/swissprot.pt
python -m tests.test_resume_merge --resume-data data/swissprot.pt
```

Policy: nothing launches on a GPU that has not run end to end somewhere. Run the job locally first:

```
SMOKE=1 JOB=smoke OUT=/tmp/smoke FS=/tmp/fs DEADLINE=$(( $(date +%s) + 3600 )) bash jobs/a10_profile/run.sh
```

## GPU jobs (Lambda Cloud, HTTPS only)

`infra/lambda_job.py` (fetched and mirrored job files land under `infra/runs/<run>/job/`) picks the first instance type with capacity from a preference list in any
allowed region (`--type a,b,c --region any`), creates a per-region filesystem if needed, launches
with a cloud-init inbox runner, uploads job and inputs through the instance's Jupyter contents API
(chunked for large files), mirrors job output from the persistent filesystem on every poll, and
terminates on completion, on error, or at `--hard-cap`. `--dry-run` prints the choice and the
worst-case cost without launching. Arm a tag-scoped dead-man's switch (an external scheduled
terminate) before launching. If the controlling machine restarts mid-job, `infra/reattach.py`
restores supervision (mirroring, termination on completion, the original cap).

```
export LAMBDA_SSH_KEY_NAME=<a key registered in your Lambda account>
python infra/lambda_job.py --run <run-id> --job-dir jobs/a10_profile --include atlas scripts tests \
  --type gpu_1x_h100_sxm5,gpu_1x_a100_sxm4,gpu_1x_a10 --region any --hard-cap 5400 \
  --upload data/swissprot_2026_03.pt cache/swissprot_2026_03.pt --mirror long/batches.jsonl --dry-run
```

## Licence

Code: Apache License 2.0 (`LICENSE`). Measurement records and derived data in `results/`: CC BY 4.0
(`results/LICENSE`). Third-party data and models (UniProtKB/Swiss-Prot, ESM-2): see `NOTICE`.
