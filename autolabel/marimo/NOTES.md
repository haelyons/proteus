# Notes: first version of the first-read notebook

Written by Claude (the autolabel session) on 2026-11-08. Codex (gpt-6-astra, xhigh, tmux session
`autolabel-marimo`) wrote REQUIREMENTS.md, DESIGN.md, `reader.py`, `first_read.py`, `check_first_read.py` and
`checks/browser_check.py`. It stopped on the ChatGPT usage limit before writing these notes, so the checks below are
mine, run on the files exactly as Codex left them.

## What I ran

    janus/.venv-gpu/bin/python autolabel/marimo/check_first_read.py
    cd autolabel/marimo && FIRST_READ_UNIT=mixed4b:431 ../../janus/.venv-gpu/bin/python first_read.py
    cd autolabel/marimo && FIRST_READ_UNIT=mixed4e:55  ../../janus/.venv-gpu/bin/python first_read.py

## What came out

- Check suite: 8 of 8 pass in about 3.5 s. They cover both units loading, bad or missing measurements never
  passing, the prediction-before-reveal lock, a failed write keeping the reveal closed, separate records per unit,
  the pair-row sign and replication, and the profile matching a separately cached stratum.
- The notebook runs top to bottom as a script for both units, exit code 0.
- Codex's browser check (`checks/browser_check.py`) failed: it timed out waiting for the text "Saved:" after saving
  a reading in the live UI. A record was written anyway: `readings/70c229d1...` (prediction, outcome and reading),
  every field marked "AUTOMATED UI CHECK; not a scientific reading". So saving works, and the confirmation message
  may not show. Not yet diagnosed.

## Not checked

- The interactive flow by eye in `marimo edit`.
- Timing of loads per unit (DESIGN.md says measured times would be here; Codex did not get to measure them).

## Leftovers

- `checks/chrome/` (79 MB) is the headless-Chrome profile from Codex's browser check. Safe to delete by hand.
- The `readings/70c229d1...` files are test records, not readings.
