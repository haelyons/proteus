# ESM first read: notes

Built by Claude (autolabel session), 2026-11-08, on the owner's request to run the first read on ESM-2 with the
proteus atlas. Budget authorised: USD 100 on Lambda.

## Pieces
- `export_firstread.py`: runs on a Lambda box with `esm-atlas-results` (us-east-1) mounted, read only. Turns an atlas
  (`top_val/top_tok`, reservoirs, edges, bin masses, moments, Gram, enrichment) plus the HF weights and the AlphaFold
  DB descriptors into `data/firstread_<model>.pt`, and the Swiss-Prot cache into `data/sequences.npz`.
  `selftest` passes; a dry run on synthetic inputs with the real 8M weights produced every field.
- `box_run.sh`: launches one gpu_1x_a10 in us-east-1 with the filesystem, runs the export for 8M (control:
  a8_shuf2) and 650M (atlas copy #2, `runs/atlas-fu650-20261104-1425/full/atlas.pt`), fetches to `data/`, terminates.
  A detached watchdog terminates the instance after 90 minutes. Logs: `box/run.log`, `box/remote.log`.
- `esm_reader.py`: data access and view rows for the notebook; the prediction outcome rules are fixed in `outcome()`.
- `first_read_esm.py`: the marimo notebook (look; family and wiring; predict; reveal; final reading). Records in
  `readings/`.

## Approximations to keep in mind
- Wiring ignores LayerNorm and attention: input and output cosines, readers by W_in(l+h) . W_out(l), and the direct
  logit effect through the tied embedding. They point at candidates; they do not measure use.
- Reservoir samples per bin are the first R by hash key, so a uniform sample of the bin (R = 16 for 8M, 8 for 650M).
- At bf16 most of a 650M unit's top-64 sits in exact ties (proteus batch_20261105: 80%); the tie rule is the smaller
  token id.
- Descriptors cover 95% of Swiss-Prot accessions; rows without a model show "–".
- The control is shuffled weights (seed 2) at 8M only.

## Runs (2026-11-08)
- Run 1 (instance b3b5d8f1, 22:37:54-22:45:53 UTC): `sequences.npz` exported (575,748 sequences, 127 annotation
  types); both bundles failed importing `transformers` (Lambda Stack's PIL lacks `Image.Resampling`). Fixed by
  upgrading pillow as the proteus jobs do, plus an import preflight.
- Run 2 (instance 9ce183f1, 22:46:59-22:57:33 UTC): `firstread_8M.pt` 37 MB (53 s), `firstread_650M.pt` 524 MB
  (121 s). Descriptor coverage of exported residues 95-96%. Both instances terminated by the script; no watchdog left.
- Spend: about 18.6 minutes of gpu_1x_a10 at USD 1.29/h, roughly USD 0.40 of the USD 100 authorised.

## Checks against proteus's published numbers (results/README.md, measurement #4 and descriptors)
- 8M L1 U86 top-64 V53 (proteus V53); U1200 V43 (V43); co-activation 0.90 (0.91); input cosine 0.81 (0.81).
- 8M L0 U797 N64 (all Asn); U1191 K59 R4 (K59 R4).
- Base secondary structure among pLDDT >= 70: loop 35.1%, helix 45.4%, strand 19.6% (proteus: 35.1 / 45.4 / 19.6).
- The notebook runs top to bottom on 8M L1 U86 and 650M L15 U2087 (about 5 s each).

## Serving
`autolabel/marimo/serve.sh` serves both notebooks (gallery at the root) on 127.0.0.1:2731, behind the Tailscale
funnel on port 8443 with the token in `~/.config/autolabel-marimo/token`.

## ESM-2 3B atlas (2026-11-09)
- New atlas: proteus `run_atlas.py`, ESM-2 3B (36 layers x 10,240 units = 368,640), Swiss-Prot 2026_03 minus the
  held-out list (194,177,125 residues), on `esm-atlas-results` at `runs/autolabel-atlas3b-20261109-1453/full/atlas.pt`
  (34.5 GB; checkpoints deleted after completion). Job: `atlas3b_job.sh`; launcher and supervisor: `box_atlas3b.sh`.
- Config as the 650M science pass (K 64, edges at pilot q 0.5..0.99999, seeded thresholds, stats v3, merge stacked)
  except: pilot 131,072 tokens (host RAM), Kr 16 and 16,384 tokens per batch (GPU memory on a 40 GB A100; 32,768 ran
  out of memory). The fast/reference equivalence gate passed on 20 batches (16 fields exact). 10 redo batches (layers
  0-1 buffer overflow; exact by construction).
- Full pass 10,660 s on gpu_1x_a100_sxm4 (us-east-1); export 432 s; bundle `data/firstread_3B.pt` 1.1 GB, descriptor
  coverage 93%. Median single-residue share of a unit's top-64 at layers 0/18/35: 1.00 / 0.36 / 0.41 (650M at 0/16/32:
  0.78 / 0.38 / 0.44).
- Launches: 1 aborted (SSH dropped right after boot; ~5 min), 2 aborted (I edited the running supervisor script, which
  crashed it and its trap terminated the box; ~11 min), 3 completed (205 min). Box spend for 3B about USD 7.3; with the
  8M/650M export, about USD 7.75 in total.
- Lessons kept in the scripts: wait for three consecutive SSH successes and retry copies; start detached jobs with
  `{ ... & }` so the SSH call returns; run supervisors from a frozen copy; Lambda Stack needs pillow upgraded for
  transformers.
