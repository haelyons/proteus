# Inventory (2026-11-11)

Read-only. Sources: Lambda API `GET /file-systems`, `/instances`, `/instance-types` through
`infra/lambda_api.py` (11 Nov 2026), and this repo's records. Nothing was mounted, so file
presence on the filesystems is **as recorded, not verified**. Nothing there is to be deleted or
overwritten; new work writes under new `runs/shard-*` directories only.

## Repo

- Branches: `main` (= `origin/main`, b6c55b2), `autolabel` (main + 1, c0661cd), `shard` (this
  work, from `autolabel`). No other branches, stashes or worktrees.
- Untracked at start: `Scaling Proteus across GPUs ... .md` (the project brief; left untracked).
- No local torch, no `data/`. Local machine: 4 cores, 15 GB RAM, GTX 1070 (not used for gates).
  A CPU venv lives in `.venv/` (excluded from git).

## Lambda: instances

None running.

## Lambda: filesystems

| Name | Region | Used | In use | Ours | Recorded contents |
| --- | --- | --- | --- | --- | --- |
| `esm-atlas-results` | us-east-1 | 162.9 GB | no | yes | `cache/swissprot_2026_03.pt`, `cache/hf`, most `runs/` (below) |
| `esm-atlas-asia-south-1` | asia-south-1 | 29.6 GB | no | yes | `cache/swissprot_2026_03.pt`, `runs/atlas-sci650-20261104-1140` |
| `esm-atlas-us-west-1` | us-west-1 | 17.0 GB | no | yes | `cache/swissprot_2026_03.pt`, `runs/atlas-implab-20261104-1000` |
| `metis-env-us-east-1` | us-east-1 | 12.6 GB | no | other project | not touched |
| `latent-prompts-260226` | us-west-2 | 7 B | no | other project | not touched |

Runs recorded on `esm-atlas-results/runs/`:

| Run | What | Instance | Record in repo |
| --- | --- | --- | --- |
| `atlas-phase1-20261103-{1538,1548,1603}` | phase 1 benchmarks | A10 | `results/phase1_a10_20261103`, manifest |
| `atlas-a10prof-20261103-1937` | forward profile, long run | A10 | `results/a10_profile_20261103`, manifest |
| `atlas-fu650-20261104-1425` | 650M full pass, atlas copy #2, gate A/B, held-out | A100 SXM4 40 GB | `results/followup_650m_20261104`, manifest |
| `atlas-desc-20261104-2130` | AlphaFold DB descriptors | A10 | `results/descriptors_20261104`, manifest |
| `atlas-wab-20261105-0140` | filtered vs `--naive` pairs (`pre_filtered`, `pre_naive`, `*_topk`) | see `results/workload_ab_20261105` | `results/workload_ab_20261105` |
| `atlas-batch-20261105-0258`, `atlas-batch2-20261105-0348` | 8M batch of ruler/control jobs | see `results/batch_20261105` | `results/batch_20261105` |
| `autolabel-atlas3b-20261109-1453` | 3B full pass, `full/atlas.pt` 34.5 GB | A100 SXM4 40 GB | `autolabel/esm/NOTES.md`, `autolabel/esm/box/atlas3b/logs` |

`results/descriptors_20261104/fs_manifest.sha256`: 62 small files (logs, json, csv) in six run
directories of `esm-atlas-results`, all dated 3–4 Nov. It holds no `atlas.pt` hash and nothing
from 5 Nov onward. The atlas files are instead pinned by per-field digests in `run.json`.

## Reference atlases and their configs

| | 650M copy #2 (gate reference) | 650M copy #1 | 3B |
| --- | --- | --- | --- |
| Path | `esm-atlas-results/runs/atlas-fu650-20261104-1425/full/atlas.pt` | `esm-atlas-asia-south-1/runs/atlas-sci650-20261104-1140/full/atlas.pt` | `esm-atlas-results/runs/autolabel-atlas3b-20261109-1453/full/atlas.pt` |
| Size | 10.85 GB | 10.85 GB | 34.5 GB |
| Layers x units | 33 x 5,120 | same | 36 x 10,240 |
| Batches / residues | 6,143 / 194,177,125 | same | 12,219 / 194,177,125 |
| tokens per batch / pilot tokens | 32,768 / 262,144 | same | 16,384 / 131,072 |
| K / Kr | 64 / 64 | same | 64 / 16 |
| Common | bf16, triton backend, stats v3, merge stacked, `--seed-thresholds`, seed 0, window 8192, edges at q 0.5 0.9 0.99 0.999 0.9999 0.99999, `--exclude-acc lists/heldout_created_after_2021-11-17.tsv` | | |
| Seconds: load / pilot / stream / save | 9.7 / 151.8 / 4,143.5 / 149.2 | 10.9 / 153.0 / 4,770.5 / 140.7 | 15.0 / 169.6 / 10,464.9 / 70.2 |
| Field digests in `run.json` | yes (9 fields) | no | yes (9 fields) |
| Extras | `--pair-every 25`, profiles at 500 and 3000, `--prefetch 2` | `--pair-every 25`, no prefetch | `--prefetch 2` |

## What to reuse

1. **650M gate reference.** `results/followup_650m_20261104/full/run.json` holds sha256 of
   `top_val, top_tok, res_tok, res_val, res_key, sub_bin, enr, ann_tot, edges`. A merged sharded
   650M atlas with the same config can be checked against those nine exact fields from the repo
   alone, with no mount. This also tests bitwise agreement with a different physical A100.
   Float sums (`sub_sum`, `sub_sq`, `gram`) need a file: use the N=1 run of the scaling benchmark.
2. **Scaling predictions.** `results/followup_650m_20261104/full/batches.jsonl` (6,143 lines:
   per-batch seconds, candidate fractions, redo layers) is the basis for per-shard time and
   filter-loss predictions.
3. **Filtered vs naive.** `results/workload_ab_20261105` (`pre_filtered` / `pre_naive` run
   records) is the existing single-GPU record for the second baseline.
4. **Data cache per region.** `cache/swissprot_2026_03.pt` is recorded in all three regions, so
   no upload is needed in us-east-1, us-west-1 or asia-south-1.
5. **Job pattern.** `jobs/followup_650m/run.sh`: `SMOKE=1` path, nvidia-smi logger, GPU tests,
   gate, deadline handling. `infra/lambda_job.py --dry-run` prints the worst-case cost.
6. **3B.** Reference for a later sharded 3B run; same nine digests are committed.

## Instance types (capacity on 11 Nov 2026; changes by the hour)

| Type | GPUs | $/h | Capacity |
| --- | --- | --- | --- |
| `gpu_1x_a10` | 1 x A10 24 GB | 1.29 | us-east-1, us-west-1 |
| `gpu_1x_a100_sxm4` | 1 x A100 40 GB SXM4 | 1.99 | asia-south-1, us-east-1, us-west-2 |
| `gpu_1x_h100_sxm5` | 1 x H100 80 GB | 4.29 | us-southeast-1, us-south-2, us-south-3 |
| `gpu_2x_h100_sxm5` | 2 x H100 80 GB | 8.38 | us-southeast-1, us-south-2, us-south-3 |
| `gpu_8x_a100` | 8 x A100 40 GB SXM4, 124 vCPU, 1.8 TB RAM | 15.92 | europe-central-1, me-west-1, us-west-1 |

Listed without capacity: `gpu_2x_a100`, `gpu_4x_a100`, `gpu_2x/4x_a6000`, `gpu_8x_v100`,
`gpu_4x/8x_h100_sxm5`, `gpu_8x_a100_80gb_sxm4`, B200 types. Full response saved outside the repo.

Consequences:

- The only way to 1, 2, 4 and 8 **identical** GPUs today is one `gpu_8x_a100` box with device
  subsets. It has the same GPU model as the 650M reference, and us-west-1 already has
  `esm-atlas-us-west-1` with the data cache.
- The two-GPU bitwise check can run on two `gpu_1x_a100_sxm4` (physically separate hosts) or
  inside the 8x box.

## Gaps

- `infra/lambda_api.py` expects the bearer token to be injected by an egress proxy. This machine
  has no such proxy; the inventory injected the key in-process from the owner's key file. The
  launcher needs the same (environment variable, never logged) before any launch.
- `infra/lambda_job.py` launches one instance per job and has no multi-GPU awareness.
- Whether `cache/hf` on `esm-atlas-us-west-1` holds the 650M weights is not recorded.
