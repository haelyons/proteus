# Random directions in shuffled-weight networks (RS1–RS3): predictions

Written 2026-11-12 with amendment 1 of `results/science_decision_rule.md`, before any RS atlas
exists and before any direction content (R or RS) from a real network is read. Measured by the
comparison job (not yet specified or built), 8M first.

## What I know and reason from

- Shuffled-weight neurons at 8M (pre-GELU hook, `results/ruler_checks_20261111/`): every sampled
  unit has a residue mode (`share_residue_mode` 100% against 96.9% trained), and their
  `primary_jsd` medians are 0.140–0.156 against 0.189 trained. Their peaks are residue-identity
  detectors.
- If a shuffled network's post-GELU vector is mostly a function of the token's residue, any linear
  read-out of it is too. A direction's peak then also selects one or two residues, so summing does
  not dilute it much: the planted checks show the loss comes from mixing parts with **different**
  states (median 0.09–0.27 against parts 0.73–0.79). Mixtures sharing a state stay close to their
  parts (0.35–0.39 against 0.39–0.63).
- The competing reading (summing dilutes any elementwise-nonlinear unit) predicts d(Sc, RSc)
  >= +0.147. I predict against it, with moderate confidence. Context mixing over six random
  attention layers is the part I cannot bound.
- Interval width: with 1,500 columns per source, the standard error of Cliff's d is about 0.02, so
  95% intervals about ±0.04.

## Predictions (8M)

| # | Quantity | Prediction | Wrong if |
| --- | --- | --- | --- |
| RS-1 | d(Sc, RSc), c = 1, 2, 3 (Cliff's d of `primary_jsd`, Sc minus RSc) | each in [-0.10, +0.12], none significant at d >= 0.147 | any d >= +0.147 or <= -0.147 |
| RS-2 | Bootstrap 95% interval width of each d(Sc, RSc) | 0.05–0.12 | outside |
| RS-3 | Spread of d(Sc, RSc) across the three controls | max minus min at most 0.10 (the shuffled controls differed by 0.06 in d(N, Sc)) | above 0.10 |
| RS-4 | `share_residue_mode` of RSc (secondary) | at least 90% for each c (directions in shuffled networks are residue read-outs too) | below 90% |

At 650M the predictions are sealed separately, after the 8M outcome is recorded.
