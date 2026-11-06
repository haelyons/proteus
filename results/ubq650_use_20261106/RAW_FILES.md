# Raw tensors for the 650M L15/2087 use study (not in git: the repository ignores *.pt and *.log)

`RAW_FILES.sha256` and `RAW_FILES.sizes` list every raw file with its checksum. Copies:
- the persistent Lambda filesystem `esm-atlas-results` (us-east-1), under `runs/atlas-use-20261106-1330/`:
  `fresh/`, `disc_iv/`, `smoke/` (written there by the job);
- the session container that produced them (ephemeral).

Discovery-pilot tensors (`discovery_pilot/*.pt`) were produced on CPU in the session container only. Their per-item
values are in `discovery_pilot/items.csv` and `discovery_pilot/intervention_items.csv`.

Per-item raw values for every readout, including every patched variant and control, are in git as CSV:
`fresh/items.csv`, `fresh/intervention_items.csv`, and the same for the discovery pilot and the GPU discovery
rerun. The `.pt` files add 2087's value at every position, the named screen units' values, and layer-15
reference-block statistics.
