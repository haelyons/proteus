# Spec: the comparison at ESM-2 650M (calibration, N, R, S1–S3, RS1–RS3, H1, H3)

Status: frozen at the commit that adds this file. The owner's decision (2026-11-12): run the
neuron-versus-direction comparison at 650M next. It implements `results/science_decision_rule.md`
(sealed, amendment 1) at 650M, including H3 (scale) against the recorded 8M result. Reuses
`scripts/compare_report.py` and `scripts/compare_sources.py` unchanged. `atlas/ruler.py` stays
byte-identical.

## Atlas config (all four runs)

```
--model 650M --data $FS/cache/swissprot_2026_03.pt --device cuda --dtype bf16 --backend triton
--tokens-per-batch 32768 --pilot-tokens 262144 --K 64 --Kr 256 --edge-q 0.5 0.9 0.99 0.999 0.9999 0.99999
--seed-thresholds --seed 0 --exclude-acc heldout_created_after_2021-11-17.tsv --prefetch 2
--stats-impl v3 --merge-impl stacked --no-gram --ckpt-every 1000000
--columns neurons_post,randdir:512:0 --source-cap neurons_post=0.04
```

- `Kr 256` matches the 8M comparison, so H3 compares like with like. The 650M reference atlas
  (4 Nov, `Kr 64`, pre-GELU) therefore has no bitwise counterpart here. Exact sharding at 650M is
  already established (`results/shard_scale_650m_20261111/`).
- `--no-gram`: the comparison reads no Gram, which costs about 10 GB per 5,120-wide source.
- Candidate cap 0.04 for `neurons_post`: bf16 ties at GELU's minimum were about 1% of a column's
  tokens at 8M (`specs/calib.md` 3b). A larger cap costs memory across 33 layers, and an overflow
  stays exact.

Runs: `trained` (no control) and `shuf<c>` (`--control shuffled --control-seed c`), c = 1, 2, 3.

## Placement

One 8-GPU host. The four runs run concurrently, two GPUs each (GPUs `2i` and `2i + 1` for run
`i`), each sharded `0/2` and `1/2` (`specs/shard.md`). Per run: a pilot on its first GPU
(`--pilot-out ... --pilot-only`), then the two shards (`--pilot-from`), then `merge_atlas.py`
per source (`shard_neurons_post.pt` and `shard_randdir_512_0.pt` separately) into
`$SCRATCH/<run>/atlas_<source>.pt`. A failed run fails the job after the others finish.

## Steps (`jobs/compare_650m/run.sh`)

Env `JOB OUT FS DEADLINE`; optional `WORKERS` (default `nproc - 2`), `PER_LAYER` (default 250).
Runner-file tolerance, `SCRATCH`, step wrapper, `timeline.json`, preflight, GPU record and the
1 Hz `nvidia-smi` logger (every GPU) as `jobs/compare_8m/run.sh`. Model download once, before any
timed GPU step, as `jobs/shard_scale/run.sh`.

1. `inputs` (data, descriptors, UniRef map, the shipped 8M `compare.json`), with sha256.
2. Four runs as above; each run's `run.json`, `batches.jsonl` and merge records into
   `$OUT/<run>/`.
3. `sample`: `compare_sources.py sample --atlas N=trained/atlas_neurons_post.pt
   R=trained/atlas_randdir_512_0.pt S<c>=shuf<c>/atlas_neurons_post.pt
   RS<c>=shuf<c>/atlas_randdir_512_0.pt --per-layer PER_LAYER --seed 0` (33 layers: 8,250 columns per
   source).
4. `score_<NAME>` for the eight sources, with the whole-build descriptors (`--workers WORKERS`).
5. `summary` (`summarise --pool N,R,S1,S2,S3,RS1,RS2,RS3`), then `report` (`compare_report.py`,
   no `--repro`) → `$OUT/compare.json`, `$OUT/compare.md`.
6. `h3`: `scripts/h3_report.py --small results_8m_compare.json --large $OUT/compare.json --out
   $OUT/h3.json` (below).
7. Copy `trained`'s two atlases to `$OUT/atlases/` and record every atlas's digests
   (`atlas.core.result_digests`) and `directions_sha256` in `$OUT/atlases/digests.json`.

`SMOKE=1`: CPU, fp32, torch backend, ESM-2 8M (not 650M) from the local HF cache, the calib
SMOKE's tiny config with `--Kr 16 --edge-q 0.3 0.5 0.7 0.9 0.95 0.99`, a synthetic whole-build
descriptor file, two "GPUs" simulated as two CPU processes per run, `PER_LAYER=4`; the shipped
8M `compare.json` stand-in is the SMOKE's own. It must exercise the pilot, both shards, both merges
per run, scoring, `report` and `h3`.

## `scripts/h3_report.py` (frozen interface)

```
python scripts/h3_report.py --small COMPARE_JSON --large COMPARE_JSON --out H3_JSON
```

For `d_NR` and for each `calibration[Sc]`: `"grows"` if the large model's interval lies strictly
above the small model's (`large.ci[0] > small.ci[1]`), `"shrinks"` if strictly below
(`large.ci[1] < small.ci[0]`), else `"flat"`.

```
{"d_NR": {"small": {"d", "ci"}, "large": {"d", "ci"}, "trend": T},
 "calibration": {"S1": {...same...}, "S2": {...}, "S3": {...}},
 "h1": {"small": V, "large": V}}
```

Worked example: small `d_NR` `{"d": -0.021, "ci": [-0.062, 0.023]}`, large `{"d": 0.10, "ci":
[0.08, 0.12]}`: 0.08 > 0.023, so `"grows"`. Large `ci = [0.02, 0.06]`: overlaps, so `"flat"`.

## Acceptance (blind tests)

- `h3_report.py`: the worked example; touching intervals give `"flat"`; `"shrinks"` mirrored.
- `SMOKE=1` end to end: `compare.json` and `h3.json` frozen keys; for each run and source the merged
  atlas equals an unsharded run of the same config (exact fields bitwise, float sums within 1e-9),
  which is the sharding invariant inside the comparison job.

## Cost

One `gpu_8x_a100` (40 GB) or `gpu_8x_a100_80gb_sxm4` in us-east-1 on `esm-atlas-results`. Estimates,
not measurements: atlases about 50 min (four 2-way runs in parallel), scoring about 60 min
(66,000 columns on about 120 workers), setup about 15 min, about 2.1 h in total. Hard cap 3.5 h:
at most $55.72 (40 GB) or $78.12 (80 GB). Launch only with the owner's yes on the dry run.
