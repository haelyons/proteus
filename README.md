# proteus

Per-neuron atlas of ESM-2 FFN units over Swiss-Prot without materialising activations.
Claim under test: an exact, monotone-threshold filter turns a ~70 TB problem (168,960 FFN
neurons x ~2e8 residues, bf16) into a one-GPU, roughly one-hour job. The fused Triton
compaction kernel is a secondary measurement. Phase-1 measurements: `results/README.md`.

## What is computed (all exact w.r.t. the stated definition)

| quantity | definition | how |
|---|---|---|
| top-K | per (layer, pole, neuron) the K largest s = +h / -h over valid tokens | running K-th value tau only rises => filter `s > tau` is exact |
| reservoir | per (layer, pole, neuron, bin) the K_r tokens with smallest key u(tok) among tokens with s in the bin | bottom-k on a bijective 32-bit hash of the packed token id; same monotone argument; no key ties |
| subsample stats | sums, sums of squares, bin masses, annotation contingency counts, Gram | exact on the hashed subsample u < rho_sub * 2^32 (estimates of full-data values; Gram is structure, not statistics) |

Bins: lower edges per (layer, pole, neuron) at pilot quantiles (0.5, 0.9, 0.99, 0.999, 0.9999)
of s. Hook point: `encoder.layer[i].intermediate.dense` output (pre-GELU). Both poles kept; the
negative pole is barely transmitted by GELU (min ~ -0.17) and is reported as secondary.

Per batch, the hooks do one compare-and-compact pass over H (threshold = min(top-K tau,
lowest under-filled reservoir edge)) plus row gathers of the hashed subsample. Row-level
decisions (valid rows, token ids, keys, subsamples) are made on the host before the forward.
Merges run once after the forward, across all layers. If a layer's candidate buffer overflows
(warm-up, bursts) the batch is re-run densely for that layer: exact either way.

State is small and merges by construction (`Atlas.merge_from`): runs over disjoint data combine
exactly; `scripts/run_atlas.py` checkpoints atomically and resumes from the batch offset.

Controls: `--control shuffled` permutes values within every weight and bias tensor (InterPLM's
control); `--control random` is a fresh init. Tokens: id = seq_idx * 1024 + residue position.
Sequences longer than 1022 residues are truncated (Swiss-Prot 2026_03: 3.21% of sequences,
5.15% of residues dropped).

## Layout

- `atlas/core.py` - Atlas (streaming state, filter, merges, checkpoint/merge), hashing
- `atlas/filter_triton.py` - fused compaction kernel (sync-free; count stays on device)
- `atlas/reference.py` - offline ground truth + pilot edges
- `atlas/data.py`, `atlas/esm.py` - Swiss-Prot batching plan, ESM-2 loading/controls/hooks
- `scripts/build_swissprot.py` - sequences + coarse residue annotations (UniProt REST)
- `scripts/run_atlas.py` - end-to-end run (checkpoints, `--resume`, `--deadline` clean stop)
- `scripts/bench_phase1.py` - forward cost vs naive top-K vs filter (torch / Triton), stage-by-stage output
- `scripts/diag_candidates.py` - observed vs iid candidate fraction under different stream orders
- `tests/` - exactness (fp32/bf16, overflow redo), Triton kernel, merge, checkpoint/resume
- `infra/` - Lambda over HTTPS only (see below)
- `jobs/phase1/run.sh` - the phase-1 GPU job
- `results/` - curated measurement records

## Run (CPU checks)

```
python -m tests.test_exactness
TRITON_INTERPRET=1 python -m tests.test_triton_kernel
python scripts/build_swissprot.py --out data/swissprot.pt
python -m tests.test_resume_merge --resume-data data/swissprot.pt
python scripts/run_atlas.py --model 8M --data data/swissprot.pt --out runs/8M --device cpu --dtype fp32 --max-seqs 500
```

## GPU jobs (Lambda Cloud, HTTPS only)

`infra/lambda_job.py` launches an instance with a cloud-init inbox runner, uploads the job and
inputs through the instance's Jupyter contents API, mirrors job output (written to the
persistent filesystem) every poll, and terminates on completion, on error, or at `--hard-cap`.
Arm an external dead-man's switch (tag-scoped terminate) before launching.

```
export LAMBDA_SSH_KEY_NAME=<a key registered in your Lambda account>
python infra/lambda_job.py --run <run-id> --job-dir jobs/phase1 --include atlas scripts tests \
  --type gpu_1x_a10 --region us-east-1 --hard-cap 5400 \
  --upload data/swissprot_2026_03.pt cache/swissprot_2026_03.pt --mirror phase1.jsonl
```
