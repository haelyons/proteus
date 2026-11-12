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
