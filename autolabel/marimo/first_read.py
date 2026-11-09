import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="First read of one unit")


@app.cell(hide_code=True)
def _():
    import os
    import sys
    from pathlib import Path

    import marimo as mo

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import reader

    return mo, os, reader


@app.cell(hide_code=True)
def _(mo, os, reader):
    unit = mo.ui.dropdown(reader.UNITS, value=os.environ.get("FIRST_READ_UNIT", reader.UNITS[0]),
                          label="InceptionV1 unit", allow_select_none=False)
    mo.vstack([
        mo.md("""# First read of one unit
        Look, explain, predict, then reveal. Keep **what it responds to** separate from **how it is wired**.
        This is a replay of existing ImageNetV2 measurements. IDs are visible and earlier views use overlapping
        splits, so this is not a blind benchmark. Record what you already know.
        Changing units starts a new reading; saved records remain in `autolabel/marimo/readings/`."""), unit])
    return (unit,)


@app.cell(hide_code=True)
def _(mo, reader, unit):
    packet = reader.load_packet(unit.value)
    session = reader.ReadSession(packet)
    get_receipt, set_receipt = mo.state(None)
    get_saved, set_saved = mo.state(None)
    get_error, set_error = mo.state("")
    return (
        get_error,
        get_receipt,
        get_saved,
        packet,
        session,
        set_error,
        set_receipt,
        set_saved,
    )


@app.cell(hide_code=True)
def _(mo, packet):
    mo.md(f"""
    ## {packet['unit']} · measurement
    Lucent InceptionV1 (`inception5h.pth`); `{packet['layer']}` output channel {packet['channel']},
    **post-ReLU spatial maximum**, float32. Input: 224×224 centre-cropped ImageNetV2 matched-frequency
    thumbnails, pixel values minus 117. Card pool C: {packet['n_card']:,} / 10,000 photos,
    `crc32(name) % 10 < 7`. Raw card maximum: **{packet['maximum']:.3f}**;
    zero responses: {packet['zero_fraction']:.1%}. This is a corpus maximum, not a possible maximum.

    Old grid captions use a 0–10 scale relative to that maximum. Raw values appear below.
    Signed negatives and input contributions use the older original-crop cache before JPEG compression.
    Sources: `janus/bank/img_acts.py:33`, `janus/tools/proploop.py:116`, `janus/tools/lookviews.py:169`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    def table(rows, **kwargs):
        return (mo.ui.table(rows, selection=None, show_column_summaries=False, show_data_types=False,
                            page_size=10, **kwargs) if rows else mo.md("Unavailable in the existing caches."))

    def cached_image(path, caption):
        return mo.image(path, width="100%", caption=caption) if path else mo.md(f"Unavailable: {caption}")

    return cached_image, table


@app.cell(hide_code=True)
def _(cached_image, mo, packet, table):
    _images = packet["images"]
    mo.vstack([mo.md("## What it responds to · observations"), mo.ui.tabs({
        "Hotspots and crops": mo.vstack([
            cached_image(_images["card"], "Twenty strongest card photos; boxes locate peak map cells."),
            cached_image(_images["crops"], f"Same photos/order, {packet['rf_side']} px effective-receptive-field crops."),
            table(packet["top"], label="Image IDs and raw values in grid order")]),
        "Across the range": mo.vstack([
            mo.md("Each row samples 60–100%, 40–60%, 20–40%, or 5–20% of the card maximum. These are amplitude bins, not quantiles."),
            cached_image(_images["range"], "Stratified exemplars; rows run from stronger to weaker."),
            cached_image(_images["random"], "Random card-pool examples, excluding the top grid."),
            cached_image(_images["nearmiss"], "Two least-active card photos for each of the top five common classes."),
            mo.md("Least-active within a class can still mean substantial activation; do not assume these are silent near misses.")]),
        "Distribution and classes": mo.vstack([
            table(packet["distribution"], label="Raw distribution over C"),
            table(packet["class_means"], label="Top class means over C; dataset names are annotations")]),
        "Negative pole": mo.vstack([
            mo.md("For 55: pre-ReLU minimum 3×3 average, cropped around the trough with blue shading. "
                  "A negative region can coexist with a positive hotspot elsewhere. Background troughs are not object labels. "
                  "For 431: signed maps are unavailable; post-ReLU silence is not a negative pole."),
            cached_image(_images.get("negative"), "Located signed negatives (raw values), only cached for mixed4e:55.")])})])
    return


@app.cell(hide_code=True)
def _(cached_image, mo, packet, reader, table):
    _peer_views = []
    for _peer in packet["peers"]:
        _tiles = []
        for _i, _photo in enumerate(_peer["photos"]):
            _caption = (f"ID {_photo} · {'target exceeds peer' if _i < len(_peer['more']) else 'peer exceeds target'}; "
                        f"target {_peer['pct_c'][_i]}%, peer {_peer['pct_p'][_i]}% of own corpus maxima")
            _tiles.append(cached_image(reader.JANUS / f"bank/img/thumbs/{_photo:05d}.jpg", _caption))
        _peer_views.append(mo.vstack([mo.md(f"**{_peer['unit']}** · class-profile Pearson r = {_peer['corr']:.3f}"),
                                     mo.hstack(_tiles, widths="equal")]))
    mo.vstack([mo.md("""## What distinguishes it from peers?
    Class-profile neighbours correlate class means over all 10,000 photos. Disagreements below use the old D split,
    with each unit scaled by its own corpus maximum and centred within class. Captions show uncentred amplitudes.
    Co-activation neighbours instead use normalized, uncentred Gram similarities on C.
    Neither ranking is a wiring family. Input-weight and output-weight families are unavailable."""),
        table(packet["gram"], label="Nearest co-activation neighbours"), *_peer_views])
    return


@app.cell(hide_code=True)
def _(cached_image, mo, packet, table):
    mo.vstack([mo.md("## How it is wired · separate evidence"), mo.md(packet["wiring_note"]),
               table(packet["wiring"], label="Input weights versus contributions; absolute-strength ranks"),
               cached_image(packet["images"].get("wiring"),
                            "Cached feature visualizations: target, signed inputs, then three positive readers. Labels give weights."),
               mo.md("Renders are optimized stimuli, not natural examples or proof of causal use. "
                     "Record a wiring hypothesis separately; 'not measured' is valid.")])
    return


@app.cell(hide_code=True)
def _(
    get_error,
    get_receipt,
    mo,
    packet,
    reader,
    session,
    set_error,
    set_receipt,
):
    def commit_prediction(fields):
        try:
            set_receipt(session.commit_and_reveal(fields))
            set_error("")
        except (OSError, ValueError) as error:
            set_error(str(error))

    prediction_form = mo.ui.dictionary({
        "reader": mo.ui.text(debounce=False, label="Reader name or initials"),
        "prior_exposure": mo.ui.text_area(debounce=False, full_width=True, label="What did you already know about this unit or these results? Write 'none' if none."),
        "response": mo.ui.text_area(debounce=False, full_width=True, label="Provisional response explanation: what distinction does it draw?"),
        "evidence": mo.ui.text_area(debounce=False, full_width=True, label="Evidence: image IDs, raw values, peer IDs, or source/table references"),
        "rival": mo.ui.text_area(debounce=False, full_width=True, label="Rival explanation; what result would separate it from yours?"),
        "confidence": mo.ui.slider(0, 1, step=0.05, value=0.5, label="Confidence in the explanation", show_value=True),
        "transform": mo.ui.dropdown(reader.TRANSFORMS, value="mirror", label="Transformation to predict", allow_select_none=False),
        "low": mo.ui.number(start=0, value=0.0, step=0.05, label="Lower retention bound (inclusive)"),
        "high": mo.ui.number(start=0, value=1.0, step=0.05, label="Upper retention bound (inclusive)"),
        "category": mo.ui.dropdown(reader.CATEGORIES, value=reader.CATEGORIES[1], label="Expected layer percentile", allow_select_none=False),
    }).form(submit_button_label="Commit prediction and reveal", validate=reader.validate_prediction,
            on_change=commit_prediction, submit_button_disabled=get_receipt() is not None or not packet["profile"])
    mo.vstack([mo.md("""## Predict before reveal
    Predict **median transformed/original activation on this unit's positive top 20 of a random 1,000 C photos**.
    Each comparison channel uses its own top 20. Low percentile means unusually low retention (more loss).
    Retention 1 means unchanged; it can exceed 1. This records a prediction of an existing measurement.
    Submission writes a local prediction file before saving or showing the outcome. It makes no semantic verdict."""),
        mo.md(f"Prediction locked: `{get_receipt()['prediction']['path']}`") if get_receipt() else prediction_form,
        mo.md(f"**Save failed:** {get_error()}") if get_error() else mo.md("")])
    return


@app.cell(hide_code=True)
def _(get_receipt):
    receipt = get_receipt()
    return (receipt,)


@app.cell(hide_code=True)
def _(cached_image, mo, packet, reader, receipt, table):
    mo.stop(receipt is None, mo.md("**Results hidden.** Commit the prediction above to reveal measurements and historical references."))
    _result = receipt["outcome"]["record"]
    _spared = packet["spared"]
    _pair_note = (f"R29 D/T: {packet['pair_counts'][0]}/{packet['pair_counts'][1]} photos; different from C/H. "
                  "All eligible pairs are shown. At least five photos per answer; d is computed after within-class centring. "
                  "Held: positive d and layer percentile ≥95. Distinctive: held and at least one peer below 50; "
                  "this does not mean unique among all peers. T-null permutes answers within class. "
                  "No row is a new test of your reading. Peer order: " + ", ".join(p['unit'] for p in packet['peers'])
                  if packet['pair_counts'] else "No R29 every-pair cache for this unit. This is missing evidence, not a failed test.")
    mo.vstack([
        mo.md("## Reveal · cached measurements"),
        table([{"status": _result["status"], "retention interval hit": _result["interval_hit"],
                "layer category hit": _result["category_hit"]}]),
        mo.md(f"Prediction SHA-256: `{receipt['prediction']['sha256']}`. The score checks this prediction only."),
        mo.ui.tabs({
            "Invariance": mo.vstack([
                table([{k: reader.number(v) if isinstance(v, float) else v for k, v in r.items()} for r in packet["profile"]]),
                mo.md("Top-20 percentile: share of finite layer retentions strictly below the target. "
                      "These P0 statistics differ from the larger pool in R10; do not compare their percentiles as identical measurements."),
                table(packet["strata"], label="Across positive activation ranks in the same P0 pool"),
                mo.md("Stratum percentiles use all layer channels in the denominator; missing values are not below. "
                      "Change is median |transformed−original| / P0 maximum. Null cells mean missing strata.")]),
            "Affected versus spared": mo.vstack([
                mo.md(f"Old selected transformation: **{_spared['tstar']}**, percentile {_spared['percentile']:.2f}. "
                      "Ten affected photos from the top 200, each matched to a spared photo on original activation. "
                      "This transform was selected from the measured profile, not from your prediction.") if _spared else mo.md("Unavailable."),
                cached_image(packet["images"]["affected"], "Most affected crops"),
                cached_image(packet["images"]["spared"], "Activation-matched spared crops, same order")]),
            "Every-pair checks": mo.vstack([mo.md(_pair_note), table(packet["pair_rows"])]),
            "Calibration and references": mo.vstack([
                table(packet["basis"], label="R32: this layer's 40 random channels versus 40 random directions"),
                mo.md("`mean_held` counts rows that hold on T, including rows below 95 on D; `null` uses shuffled answers. "
                      "Many answer pairs are tested, so some null successes are expected. This is instrument context, not a cleanliness score. "
                      "Source: `janus/README.md:1043` and `janus/runs/r32/report.txt`."),
                mo.md("**431 reference:** right-facing dog heads; blind facing labels and mirror reversal support it "
                      "(`janus/README.md:618`). **55 reference:** contested; the essay's V reading has no truth class "
                      "in Janus (`janus/CONTRACT.md:43`). Rotation sensitivity fits both upright faces and a downward V "
                      "(`janus/README.md:253`). A feature visualization or successful prediction does not settle this.")])})])
    return


@app.cell(hide_code=True)
def _(get_saved, mo, reader, receipt, session, set_saved):
    mo.stop(receipt is None)

    def save_revision(fields):
        try:
            set_saved({"ok": True, **session.save_reading(fields)})
        except (OSError, ValueError) as error:
            set_saved({"ok": False, "error": str(error)})

    reading_form = mo.ui.dictionary({
        "response": mo.ui.text_area(debounce=False, full_width=True, label="Revised response explanation"),
        "wiring": mo.ui.text_area(debounce=False, full_width=True, label="Wiring interpretation and limits; 'not measured' is valid"),
        "family": mo.ui.text_area(debounce=False, full_width=True, label="What belongs to the family, and what distinguishes this unit?"),
        "failures": mo.ui.text_area(debounce=False, full_width=True, label="What failed or changed after reveal?"),
        "evidence": mo.ui.text_area(debounce=False, full_width=True, label="Evidence IDs/table rows supporting the revised reading"),
        "missing_evidence": mo.ui.text_area(debounce=False, full_width=True, label="Unresolved evidence; what is still unknown?"),
        "next_test": mo.ui.text_area(debounce=False, full_width=True, label="Next discriminating test and the result that would change your mind"),
        "class": mo.ui.dropdown(reader.CLASSES, value="cannot_tell", label="Provisional class, not ground truth", allow_select_none=False),
        "confidence": mo.ui.slider(0, 1, step=0.05, value=0.5, label="Revised confidence", show_value=True),
    }).form(submit_button_label="Save revised reading", validate=reader.validate_reading, on_change=save_revision,
            submit_button_disabled=bool(get_saved() and get_saved().get("ok")))
    mo.vstack([mo.md("## Record the reading"),
        mo.md(f"Saved: `{get_saved()['path']}`") if get_saved() and get_saved().get("ok") else reading_form,
        mo.md(f"**Save failed:** {get_saved()['error']}") if get_saved() and not get_saved().get("ok") else mo.md("")])
    return


@app.cell(hide_code=True)
def _(mo, packet, table):
    mo.accordion({"Sources and missing views": mo.vstack([
        mo.md("Paths are relative to `/home/hal/dev/interp`. Hashes are stored with committed predictions. "
              "Opening the notebook writes no reading and runs no model."),
        table([{"path": path, **info} for path, info in packet["manifest"].items()]),
        mo.md("Missing optional cache files:\n\n" + ("\n".join(f"- `{path}`" for path in packet["missing"]) or "None."))])})
    return


if __name__ == "__main__":
    app.run()
