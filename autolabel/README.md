# autolabel

What it would mean to apply Cammarata's "Neural Networks Are Already Clean" (Goodfire, 2026-10-15) to automated unit
labelling, and a first-read notebook for doing it by hand on InceptionV1 and on ESM-2 using the atlases in this repo.
Written 2026-11-08/09 in `~/dev/interp/autolabel` and moved here on 2026-11-09 (the old path is a symlink).

- `BRIEF.md`, `LAUNCH_PROMPT.md`: the owner's brief and the session prompt.
- `PREDICTIONS.md` (sealed before reading, scored after), `OBSERVATIONS.md` (evidence with citations), `ANSWERS.md`
  (the five questions discussed). Citations point into the essay and janus (github.com/haelyons/janus).
- `marimo/`: the InceptionV1 first-read notebook (`first_read.py`, loader `reader.py`), replaying janus's cached
  measurements; written by Codex from `PROMPT.md`, with `REQUIREMENTS.md`, `DESIGN.md`, `NOTES.md` and checks.
  `serve.sh` serves both notebooks in app mode behind a token.
- `esm/`: the ESM-2 first-read notebook (`first_read_esm.py`, loader `esm_reader.py`) for 8M, 650M and 3B.
  `export_firstread.py` turns an atlas plus weights and AlphaFold DB descriptors into a bundle; `box_run.sh` exports
  the 8M and 650M atlases on a Lambda box; `atlas3b_job.sh` and `box_atlas3b.sh` compute the ESM-2 3B atlas
  (`esm-atlas-results:runs/autolabel-atlas3b-20261109-1453/`) and export it. `NOTES.md` has the runs, checks and spend;
  `box/` keeps their logs.

Not committed (see `.gitignore`): the bundles in `esm/data/` (1.8 GB; rebuild with the box scripts), notebook readings,
the owner's private notes in `context/`, and machine-specific files. Run the notebooks with janus's environment:
`marimo edit autolabel/esm/first_read_esm.py`.
