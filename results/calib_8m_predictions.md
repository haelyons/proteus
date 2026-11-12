# 8M calibration atlases and the planted-shape rerun: predictions

Written 2026-11-11, before either run. Rule: `results/science_decision_rule.md` (sealed).

## Planted-shape rerun (`jobs/ruler_checks/run.sh`, `SHAPES="randdir sae:0.001" POOLING=0`)

Predictions 3 (sae:0.001 part), 4 (randdir), 5 (noise null) and 18 (mixtures not above parts) of
`results/ruler_checks_predictions.md` apply unchanged to these two shapes. Added:

| # | Quantity | Prediction | Wrong if |
| --- | --- | --- | --- |
| R1 | Both shapes finish within the 4 h cap with the vectorised generator and twice the threads per shape | yes | either times out |
| R2 | randdir, v1.2 gate | every graded type within the recorded three draws' range; noise and alpha = 0 at most 3 of 542 called | outside |
| R3 | sae:0.001, peak-level calls identical to gauss | at least 95% of columns (the 0.01 rate gave 98.3%) | below 95% |

## Calibration job (`jobs/calib_8m/run.sh`, one `gpu_1x_a100_sxm4`)

| # | Quantity | Prediction | Wrong if |
| --- | --- | --- | --- |
| C1 | `neurons_pre` from the multi-source runs vs the recorded 8M atlases (`a8_pre`, `a8_shuf1..3`): 9 exact fields | bitwise equal for all four (same config, same GPU type as recorded) | any field differs in any run |
| C2 | Calibration, post-GELU, d(trained, shuf c) for c = 1, 2, 3 | all three >= 0.147 with p < 0.01: calibration passes | any fails |
| C3 | Size of d(trained, shuf c) post-GELU | within 0.05 of the pre-GELU values (+0.225, +0.202, +0.167): the positive pole's peak bins hold the same tokens wherever GELU is increasing | any outside |
| C4 | Records of the sampled trained columns, post-GELU vs pre-GELU (same sample indices) | `score_sha256` identical for at least 60% (bf16 GELU outputs create new ties near bin edges, so not all) | below 60% |

The random-direction atlas is built and its digests recorded; nothing about its content is read.

## Addendum before launch (2026-11-11, after a local profile; no run yet)

A one-thread local profile of the `randdir` build (1,490 columns) found serial `torch.randn`
about 0.7 ms per token and GELU about 0.5 ms per token: about 10 thread-hours for the 24.4M
tokens of the design set and its pilot. Vectorising removes only Python overhead (Codex measured
3.78 s against 3.00 s for the looped version on a small set, so it is no faster). R1 as written
would have been wrong for `randdir`. The build now also computes protein blocks in parallel worker
processes (`specs/calib.md` amendment 1b, bitwise equal), 14 per shape on the 30-vCPU A10.

| # | Quantity | Prediction | Wrong if |
| --- | --- | --- | --- |
| R1' | Build step times (replaces R1's reason; R1 stands as written) | `randdir` build 40–120 min; `sae:0.001` build 60–200 min (the recorded `sae:0.01` build took 3.2 h at 4 threads with 7 shapes sharing the host; its cost is the atlas, not generation) | outside |

## Addendum 2 before the calibration relaunch (2026-11-12, no calibration result yet)

The first calibration launch (`calib-8m-20261111-2350`, one `gpu_1x_a100_sxm4`) was terminated by
the orchestrating agent at `trained` batch 4,415 of 6,143 (about 1.2 billed hours, about $2.40). No
atlas, score or calibration output was produced or read. `neurons_post` overflowed on every
layer-batch (bf16 ties at GELU's minimum; `specs/calib.md` 3b), making the job about 4 h against
its 3 h cap. Relaunch with `--source-cap neurons_post=0.1`. C1-C4 stand. Added:

| # | Quantity | Prediction | Wrong if |
| --- | --- | --- | --- |
| C5 | `neurons_post` redo layer-batches in `trained` with the cap | at most 5% of 36,858 (6,143 batches x 6 layers) | above |
| C6 | `trained` step time (three sources) | 15-45 min (recorded single-source stream 8.2 min; first launch 0.77 s per batch with the redo) | outside |

## Outcome: planted-shape rerun (2026-11-12, `results/ruler_checks_rerun_20261112/`)

Run `ruler-rerun-20261111-2349`: one `gpu_1x_a10` in us-east-1 on `esm-atlas-results`, healthy
23:55:03Z, job done 02:17:14Z, terminated 02:18:56Z (about 2.4 billed hours, about $3.10). Inputs'
sha256 equal to the first checks run's. Numbers from `timeline.json`, the build logs' final
lines, `checks.json`, `calls.json` and the separation gate rerun over all seven shapes
(`separation_all7.json`; its `stats` equal the recorded `separation.json`'s).

| # | Predicted | Measured | Held? |
| --- | --- | --- | --- |
| R1 | both shapes finish within the 4 h cap | yes: job done 2 h 22 min after start | yes |
| R1' | `randdir` build 40–120 min; `sae:0.001` build 60–200 min | `randdir` 38.5 min (2,109 s streaming, 9 redo layer-batches of 623); `sae:0.001` 139.7 min (8,271 s; dense on every batch: 50.4G of 61.3G elements were top-K candidates, the zero mass) | no (`randdir` faster than the range) |
| R2 | `randdir` gate within the recorded range; at most 3 of 542 noise and alpha = 0 called | `narrow` 3/12 (recorded 8–10/12); every other type 12/12; 5 of 542 called | no |
| R3 | `sae:0.001` peak-level calls identical to `gauss` for at least 95% | 1,464/1,490 (98.26%) | yes |

Earlier predictions applied to these shapes (`results/ruler_checks_predictions.md`): #3 (`sae:0.001`
at least 99% identical): no, 98.26%; persistence differs for 244/602 (40.5%) but "peak only" fell
194 -> 106. #4 (`randdir` within the recorded range): no (`narrow` 3/12). #5 (noise null at most 2%,
upper bound below 3%): yes for both, `randdir` 6/518 = 1.16% [0.43%, 2.50%], `sae:0.001` 3/518 =
0.58% [0.12%, 1.68%]. #18 (no mixture above its largest part): yes for both (and the five earlier
shapes, recomputed). `randdir` peak calls equal `gauss`'s for 831/1,490 (55.8%) columns.

Consequence for the qualification: unchanged. `primary_jsd` and `share_residue_mode` still
qualify on all seven shapes; `share_struct` does not. The ruler loses most `narrow` columns
when the noise has the random-direction shape rather than Gaussian (cause not measured).

## Outcome: 8M calibration (2026-11-12, `results/calib_8m_20261112/`)

Run `calib-8m-20261112-1018`: `gpu_1x_a100_sxm4` in us-east-1 on `esm-atlas-results`; the box was an
**A100-SXM4-80GB** (driver 570.148.08) although the type is listed as "A100 (40 GB SXM4)". Healthy
10:22:55Z, job done 11:40:24Z, terminated 11:43Z (about 1.34 billed hours, about $2.67; with the
first launch, about $5.10 for calibration). Config `specs/calib.md` 3 with 3b
(`--source-cap neurons_post=0.1`). Numbers from `calib.json`, `calibration.json`, `repro_*.json`,
`timeline.json`, the runs' `run.json`, and a recomputation of Cliff's d from
`pooling/scores/*.jsonl`, which equals `calibration.json` to six decimals.

| # | Predicted | Measured | Held? |
| --- | --- | --- | --- |
| C1 | `neurons_pre` bitwise equal to the recorded `a8_pre`, `a8_shuf1..3` | all four differ, slightly. `edges`, `ann_tot` and `n_seen` (194,177,125) are equal; trained: `top_val` 8 and `top_tok` 44 elements, reservoirs about 10k of about 24M entries; float sums within 5e-8 relative. The premise was wrong: the recorded 8M atlases were built on an **A10** (`results/README.md`, job batch_20261105: "no A100/H100 anywhere at 02:55"), not an A100. A GPU architecture change (sm_86 to sm_80) changing bf16 kernels fits the pattern but is not measured | no |
| C2 | calibration passes: d >= 0.147 and p < 0.01 for all three | d(trained, shuf c) = +0.236, +0.214, +0.198; p = 1.7e-28, 1.0e-23, 2.2e-20 | **yes: calibration passes** |
| C3 | post-GELU d within 0.05 of pre-GELU (+0.225, +0.202, +0.167) | differences +0.011, +0.012, +0.031 | yes |
| C4 | at least 60% of sampled trained records `score_sha256` equal post vs pre | 675/1,500 (45.0%); the pre-GELU reference is the recorded A10 atlas's scores (`results/ruler_checks_20261111/`), so C1's differences are mixed in | no |
| C5 | `neurons_post` redo layer-batches in `trained` at most 5% of 36,858 | 4 (0.01%); `neurons_pre` 0, `randdir_512_0` 22 | yes |
| C6 | `trained` step 15–45 min | 14.0 min (shuffled runs 12.7–12.9 min, two sources each) | no (faster) |

Non-insufficient columns scored: trained 1,499 of 1,500; shuffled 1,427, 1,433, 1,420. The
`randdir_512_0` atlas of `trained` (`atlases/` on the filesystem, with `atlas_neurons_post.pt`)
has its digests and `directions_sha256` in `calib.json`; nothing else about it was read.

Consequence (sealed rule, amendment 1): calibration holds at 8M at the post-GELU hook, so the
comparison may be read once built. Per amendment 1 it needs RS1–RS3, which the comparison job
builds. Its spec comes first.
