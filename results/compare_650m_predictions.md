# 650M comparison (N, R, S1–S3, RS1–RS3; H1, H3): predictions

Written 2026-11-12, before `jobs/compare_650m/run.sh` or `scripts/h3_report.py` exists and before
any 650M post-GELU or direction content is read. Spec `specs/compare_650m.md` (frozen, d9a9d47);
rule `results/science_decision_rule.md` with amendment 1. Owner's decision: 650M next.

## What I know before writing

- 8M (`results/compare_8m_20261112/`): d(N, Sc) = +0.236, +0.214, +0.198; d(N, R) = -0.021,
  95% interval [-0.062, +0.023], verdict "against"; d(Sc, RSc) = -0.047, +0.011, -0.028.
  `share_residue_mode` R 99.4% above N 96.9%. My 8M prediction P1 (+0.15 to +0.45) was wrong by a
  wide margin; these ranges are wider and centred lower for that reason.
- 650M sharding (`results/shard_scale_650m_20261111/`, one source `neurons_pre`, `Kr 64`, with
  Gram): 2-way shard stream 1,864 s, run wall 2,124 s; per shard `max_memory_allocated` 22.6 GB,
  host RSS 20 GB; dense redo on 52 of about 101,000 layer-batches. K-sweep: `Kr 256` costs about 5%
  per batch over `Kr 64`.
- 8M scoring: 1,500 columns per source in about 330 s on 28 workers, about 6 worker-seconds per
  column. Report (bootstrap, 2,000 draws, rank-based Cliff's d): 127 s at n = 1,500.

## Reasoning I commit to

- Calibration: training puts more structure into a 33-layer, 5,120-wide network than into 8M, and
  a shuffled 650M stays a residue read-out. d(N, Sc) should be at least as large as at 8M.
- d(N, R): at 8M the basis carried no advantage. At 650M, neurons are sparser after GELU across a
  4x wider layer, so a random direction mixes more units and its peaks fall back on the dominant
  residue signal. I expect a small positive gap, below the rule's 0.147 threshold. Verdict stays
  "against". The 8M interval's upper end is +0.023 and the 650M interval is about 0.036 wide
  (n = 8,250), so a gap near +0.07 reads as "grows" in H3.
- Shuffled networks: neurons and directions are both residue read-outs, as at 8M, so
  d(Sc, RSc) stays near zero.

## Predictions

| # | Quantity | Prediction | Wrong if |
| --- | --- | --- | --- |
| Q1 | Calibration (all three d(N, Sc) >= 0.147, p < 0.01) | passes | any fails |
| Q2 | d(N, Sc), each c | +0.15 to +0.45 | any outside |
| Q3 | d(N, R) | -0.05 to +0.20 | outside |
| Q4 | H1 verdict | "against" | any other |
| Q5 | Width of the 95% interval of d(N, R) | 0.025–0.050 | outside |
| Q6 | H3 trend of d(N, R) | "grows" | "flat" or "shrinks" |
| Q7 | H3 trend of calibration d(N, Sc) | no "shrinks" for any c | any "shrinks" |
| Q8 | d(Sc, RSc), each c | -0.10 to +0.10 | any outside |
| Q9 | `share_residue_mode` | N below its 8M value (96.9%); R above N, intervals apart | either fails |
| Q10 | `neurons_post` dense redo (cap 0.04) | at most 1% of each shard's layer-batches | above, in any shard |
| Q11 | Each run, pilot start to both merges done | 25–50 min | any outside |
| Q12 | Per shard process: `max_memory_allocated`; host RSS | at most 25 GB; at most 30 GB | either above |
| Q13 | `directions_sha256` | identical across the four runs | any differs |
| Q14 | Each `score_<NAME>` step | 4–12 min | any outside |
| Q15 | `report` step | 5–25 min | outside |
| Q16 | Job wall, `preflight` start to `copy_atlases` done | 1.6–2.8 h, inside the 3.5 h cap | outside |

Statistics come from `compare.json` and `h3.json`. Times come from `timeline.json` and each run's
`run.json`. Memory comes from `run.json` (`max_memory_allocated`, `peak_rss_bytes`). Redo counts
come from `batches.jsonl` (`redo_layers`).
