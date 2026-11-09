# First version

One notebook, two units, a small read-only adapter, and local JSON records. The unit picker shows IDs without supplied labels. Historical reference readings appear only after the prediction. This follows the response/wiring separation in [REQUIREMENTS R1–R10](REQUIREMENTS.md) and [ANSWERS:47](../ANSWERS.md:47).

## Data and cost

`reader.py` reads existing files directly. It never calls a Janus experiment's main function or its cache-building wrappers. It disables bytecode writes before importing Janus helpers. Paths resolve from the adapter, so launching elsewhere works.

| Data | Work at open or unit change |
|---|---|
| `rangeview/<unit>` values, receptive field and grids; `bank/img/names.json` | Load small arrays and JPEGs; summarize the 7,038-photo card pool. Preserve cached grids instead of rebuilding their selection rules. |
| `proploop/values_<layer>.npy` | Memory-map 20–32 MB; compute raw class means and three co-activation neighbours using only the target row of the Gram matrix. CPU work, about 4–6 million multiply-adds. |
| `lookviews/profile_<layer>.npz`, `rangeprofile/<layer>.csv` | Decompress 19–31 MB and run existing retention helpers on CPU. Load lower-rank statistics from the CSV. No new transforms or forwards. |
| `contrastloop`, `sparedview`, `pairtable`, `r32` | Load existing IDs, images and tables. Preserve experiment-specific labels and sample counts. |
| `circuit/circuit.npz`, signed maps, and `featvis` for 55 | CPU reductions over cached contributions; show cached renders and negative grid. No model load or optimization. |

Source SHA-256 hashes bind a saved prediction to its packet, including the notebook and adapter. Session-level caching avoids rereading large arrays on each widget change. This is seconds of CPU and disk work, zero model calls, zero GPU time and zero network calls; measured times are in `NOTES.md`. Cache generation costs are deliberately absent: layer-wide measurements are already the shared barrier described in [ANSWERS:109](../ANSWERS.md:109). Profile pool and preprocessing differences stay visible ([README R10:230](../../janus/README.md:230), [lookviews.py:3](../../janus/tools/lookviews.py:3)).

## Cells and reader flow

1. **Choose and identify.** Unit ID, exact measurement, provenance, missing-cache messages and replay limitation.
2. **Look.** Tabs for whole-image hotspots/crops, range/random/least-active examples, raw distribution and class means, signed negatives, peers and disagreements. Tables retain image IDs and raw values beside legacy normalized grid captions.
3. **Inspect wiring.** Separate input-weight/contribution table and input/reader renders for 55. Missing wiring for 431 is visible. “Not measured” is a valid wiring reading.
4. **Commit a prediction.** A form asks for the provisional explanation, evidence, rival and prior exposure. Choose any cached transformation; predict its top-20 retention interval and whether its layer percentile is low (≤20), middle (20–80 exclusive), or high (≥80). The form validates required text and bounds. It saves once and locks for this unit session.
5. **Reveal.** Only a successful prediction write unlocks the profile, lower-rank table, affected/spared view, old every-pair checks, basis summary and historical references. A separate outcome file records the measured value, interval hit and category hit. No automatic semantic verdict. Low retention means more loss, not less; enhancement can exceed one.
6. **Revise and save.** A final form keeps response, wiring and family claims separate; asks for failed predictions, missing evidence and next test; records uncertainty and a provisional class. Save once. Switching units starts a new session; old records remain on disk.

The notebook uses marimo state and form callbacks so ordinary reactive reruns do not create records. Files use a UUID session prefix and exclusive creation: `<id>.prediction.json`, `<id>.outcome.json`, `<id>.reading.json` in `readings/`. Each later file includes the prediction digest. The outcome is written after the prediction and before display. A failed save leaves the reveal closed. These files are locally editable, so the digest supplies traceability, not tamper-proof evidence.

## What stays out

No agent, API, new image synthesis, live GPU probe, corpus collection, generalized circuit explorer, SAE, automatic label acceptance, benchmark or export service. No OLMo in this pass. No new planted-control run; the existing R32 control is displayed with its multiple-comparison limitation ([README:1058](../../janus/README.md:1058)). No claim that the old T split is unseen by this reader: the notebook combines views from different rounds. A future experiment must reserve its own examples before displaying anything.

The every-pair table exists for 431 only; wiring and signed negatives exist for 55 only. A first read should expose these asymmetries, not spend compute hiding them. The remaining task for the reader is the next discriminating question, especially outside the standing battery's axes ([ANSWERS:118](../ANSWERS.md:118)).
