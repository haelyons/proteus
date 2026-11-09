import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="First read of one ESM unit")


@app.cell(hide_code=True)
def _():
    import os
    import sys
    from pathlib import Path

    import marimo as mo

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import esm_reader as er

    return er, mo, os


@app.cell(hide_code=True)
def _(er, mo, os):
    _models = er.models()
    model = mo.ui.dropdown(_models, value=os.environ.get("ESM_UNIT_MODEL", "8M" if "8M" in _models else _models[0]), label="Model")
    studied = mo.ui.dropdown(["manual"] + [k for k, v in er.STUDIED.items() if v[0] in _models], value="manual",
                             label="Or a unit studied in proteus (position only)")
    mo.vstack([mo.md("""# First read of one ESM-2 unit
    Look, then predict, then reveal. Keep **what the unit responds to** apart from **how it is wired**.
    Data: the proteus atlas over Swiss-Prot 2026_03 (194M residues; pre-GELU FFN activation), exported by
    `autolabel/esm/export_firstread.py`. The atlas is exact top-64 per unit and pole, plus uniform samples of each
    activation range. Not blind: unit positions are visible. Records go to `autolabel/esm/readings/`."""),
               mo.hstack([model, studied], justify="start")])
    return model, studied


@app.cell(hide_code=True)
def _(er, mo, model, os):
    _m = er.bundle(model.value)["meta"]
    layer = mo.ui.number(0, _m["L"] - 1, value=min(int(os.environ.get("ESM_UNIT_LAYER", 1)), _m["L"] - 1), label=f"Layer (0-{_m['L'] - 1})")
    unit = mo.ui.number(0, _m["D"] - 1, value=min(int(os.environ.get("ESM_UNIT_INDEX", 86)), _m["D"] - 1), label=f"Unit (0-{_m['D'] - 1})")
    pole = mo.ui.radio(["positive", "negative"], value="positive", label="Pole", inline=True)
    mo.hstack([layer, unit, pole], justify="start")
    return layer, pole, unit


@app.cell(hide_code=True)
def _(er, layer, mo, model, pole, studied, unit):
    if studied.value != "manual":
        M, L, U = er.STUDIED[studied.value]
    else:
        M, L, U = model.value, int(layer.value), int(unit.value)
    P = er.POLES[pole.value]
    meta = er.bundle(M)["meta"]
    get_result, set_result = mo.state(None)
    get_saved, set_saved = mo.state(None)
    mo.md(f"""## ESM-2 {M} · layer {L} · unit {U} · {pole.value} pole
    Activation: `encoder.layer[{L}].intermediate.dense` output (pre-GELU), bf16 atlas over {meta['n_seen']:,} residues;
    moments, Gram and enrichment on a hashed {meta['n_sub']:,}-residue subsample. Bins are the unit's own pilot
    quantiles {meta['bin_quantiles']}. Bundle `{er.digest(M)}`.""")
    return L, M, P, U, get_result, get_saved, meta, set_result, set_saved


@app.cell(hide_code=True)
def _(L, M, P, U, er, mo):
    _top = er.top_rows(M, L, U, P)
    _other = er.top_rows(M, L, U, 1 - P, n=24)
    _comp = er.composition([r for r in er.bundle(M)["top_tok"][L, P, U].tolist()])
    _conc = er.concentration_pct(M, L, U, P)
    _range = er.range_rows(M, L, U, P)
    _ctl = "" if _conc["control_median"] is None else f" Shuffled-weight control layer median {_conc['control_median']}; this unit is at the {_conc['pct_vs_control']}th percentile of the control."
    mo.vstack([
        mo.md(f"""## 1. Look: what it responds to
    Top-64 residues: **{_comp['residues']}** across {_comp['proteins']} proteins. The most common residue is
    {_conc['max_share']:.0%} of the list, the {_conc['pct_in_layer']}th percentile of this layer (median {_conc['layer_median']}).{_ctl}
    A list that is one residue type across many proteins may be a token detector; check the range below before deciding."""),
        mo.ui.tabs({
            "strongest (this pole)": mo.ui.table(_top, selection=None, page_size=16),
            "activation range": mo.ui.table(_range, selection=None),
            "strongest (other pole)": mo.ui.table(_other, selection=None, page_size=12),
        }),
    ])
    return


@app.cell(hide_code=True)
def _(L, M, P, U, er, mo):
    _fam = er.family_rows(M, L, U, P)
    _w = er.wiring(M, L, U)
    mo.vstack([
        mo.md("""## 2. Family and wiring: how it is built and read
    Peers in the same layer by co-activation (Pearson r on the subsample) and by input weights (cosine of what they
    read). A reading that also fits the peers describes the family; look for the one axis where this unit differs."""),
        mo.ui.tabs({k: mo.ui.table(v, selection=None) for k, v in _fam.items()}),
        mo.md(f"""**What it writes** (direct path to the amino-acid logits through the tied embedding; LayerNorm and the
    lm_head transform ignored): promotes {_w['writes']['promotes']}; suppresses {_w['writes']['suppresses']}.
    **Readers** in the next two layers (cosine of their input weights with this unit's output weights, residual path only):"""),
        mo.ui.table(_w["readers"], selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(L, M, P, U, er, get_result, mo, set_result):
    def _submit(v):
        if get_result() is not None or not v:
            return
        pred = {k: v[k] for k in er.PRED_CHOICES} | {"reading": v["reading"], "evidence": v["evidence"], "confidence": v["confidence"]}
        set_result(er.save_prediction(M, L, U, P, pred))

    _fields = mo.ui.dictionary({
        "reading": mo.ui.text_area(label="Your reading: what one thing might this unit respond to?", full_width=True),
        "evidence": mo.ui.text_area(label="Evidence from the views above, and the main rival reading", full_width=True),
        **{k: mo.ui.dropdown(opts, value=opts[-1], label=f"Predicted {k} of the top-64 against the base rate") for k, opts in er.PRED_CHOICES.items()},
        "confidence": mo.ui.slider(0.0, 1.0, step=0.05, value=0.5, label="Confidence"),
    })
    pred_form = mo.ui.form(_fields, submit_button_label="Commit prediction (saves, then reveals)", on_change=_submit,
                           validate=lambda v: None if v and v["reading"].strip() else "Write a reading first.")
    mo.vstack([mo.md("## 3. Predict before the reveal"), pred_form]) if get_result() is None else mo.md("## 3. Prediction committed")
    return


@app.cell(hide_code=True)
def _(L, M, P, U, er, get_result, mo):
    _r = get_result()
    if _r is None:
        _out = mo.md("_The structure profile and annotation enrichment appear after you commit a prediction._")
    else:
        _p = _r["profile"]
        _hits = ", ".join(f"{k}: predicted {'✓' if ok else '✗'} (observed {_r['observed'][k]})" for k, ok in _r["hits"].items())
        _out = mo.vstack([
            mo.md(f"""## 4. Reveal
    Saved session `{_r['session']}`. {_hits}.
    Rules (fixed in `esm_reader.outcome`): a secondary-structure class counts if its share among confident residues
    (pLDDT >= 70) exceeds the base rate by >= 0.15; buried/exposed by mean burial against base quantiles;
    disordered if the pLDDT < 50 share exceeds the base by >= 0.2."""),
            mo.ui.table([{"set": "top-64", **_p["top"]}, {"set": "base rate (all residues)", **{k: v for k, v in _p["base"].items() if k != "burial_q10_50_90"}}], selection=None),
            mo.md("Across the activation range (reservoir samples per bin):"),
            mo.ui.table(_p["bins"], selection=None),
            mo.md("Swiss-Prot annotation enrichment above each level (subsample counts; ratio against the base rate):"),
            mo.ui.table(er.enrichment_rows(M, L, U, P) or [{"note": "no enrichment counts in this atlas"}], selection=None, page_size=10),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(er, get_result, get_saved, mo, set_saved):
    _r = get_result()

    def _final(v):
        if get_saved() is None and v and _r is not None:
            set_saved(er.save_reading(_r["session"], v))

    if _r is None:
        _o = mo.md("")
    elif get_saved() is not None:
        _o = mo.md(f"## 5. Reading saved: `{get_saved()}`")
    else:
        _o = mo.vstack([mo.md("## 5. Final reading"), mo.ui.form(mo.ui.dictionary({
            "responds_to": mo.ui.text_area(label="What it responds to", full_width=True),
            "wiring": mo.ui.text_area(label="How it is wired (family, writes, readers), or 'not established'", full_width=True),
            "family_difference": mo.ui.text_area(label="The axis on which it differs from its peers", full_width=True),
            "class": mo.ui.dropdown(["one", "one_plus_related", "unrelated", "cannot_tell"], value="cannot_tell", label="Class"),
            "failed": mo.ui.text_area(label="Predictions that failed, and what that changes", full_width=True),
            "next_test": mo.ui.text_area(label="The next test that could break this reading", full_width=True),
        }), submit_button_label="Save reading", on_change=_final)])
    _o
    return


if __name__ == "__main__":
    app.run()
