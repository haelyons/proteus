"""Data access for the ESM first-read notebook (first_read_esm.py).

Reads the bundles written by export_firstread.py (data/firstread_<model>.pt, data/sequences.npz) and turns one
unit's pole into plain rows for marimo tables. No model is loaded and nothing is computed on a GPU.
Outcome rules for the prediction step are fixed here, before any reading: see outcome().
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("ESM_FIRSTREAD_DATA", HERE / "data"))
READINGS = Path(os.environ.get("ESM_FIRSTREAD_READINGS", HERE / "readings"))
SS3 = ("loop", "helix", "strand")
POLES = {"positive": 0, "negative": 1}
# units studied in proteus (results/README.md); listed by position only, so the label does not prime the reader
STUDIED = {"8M L1 U86": ("8M", 1, 86), "8M L1 U1200": ("8M", 1, 1200), "8M L3 U1238": ("8M", 3, 1238),
           "8M L5 U668": ("8M", 5, 668), "8M L0 U789": ("8M", 0, 789), "8M L0 U767": ("8M", 0, 767),
           "8M L5 U1181": ("8M", 5, 1181), "8M L1 U746": ("8M", 1, 746), "650M L15 U2087": ("650M", 15, 2087)}
_B, _S = {}, {}


def models() -> list[str]:
    return sorted((p.stem.split("_", 1)[1] for p in DATA.glob("firstread_*.pt")), key=lambda m: float(m[:-1]) * (1e3 if m.endswith("B") else 1))


def bundle(model: str) -> dict:
    if model not in _B:
        _B[model] = torch.load(DATA / f"firstread_{model}.pt", map_location="cpu", weights_only=False, mmap=True)
    return _B[model]


def seqs() -> dict:
    if not _S:
        z = np.load(DATA / "sequences.npz", allow_pickle=False)
        _S.update(blob=z["blob"], off=z["off"], acc=z["acc"], vocab=[str(v) for v in z["vocab"]])
    return _S


def digest(model: str) -> str:
    p = DATA / f"firstread_{model}.pt"
    st = p.stat()
    return hashlib.sha256(f"{p.name}:{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:16]


def window(tok: int, flank: int = 12) -> dict:
    s = seqs()
    q, p = divmod(int(tok), int(bundle_meta_stride()))
    a, b = int(s["off"][q]), int(s["off"][q + 1])
    seq = s["blob"][a:b].tobytes().decode()
    left, c, right = seq[max(0, p - flank):p], (seq[p] if p < len(seq) else "?"), seq[p + 1:p + 1 + flank]
    return {"protein": str(s["acc"][q]), "pos": p + 1, "residue": c,
            "context": ("…" if p - flank > 0 else "") + left + "[" + c + "]" + right + ("…" if p + 1 + flank < len(seq) else "")}


def bundle_meta_stride() -> int:
    return 1024


def _ss3(v: int) -> str:
    return SS3[v] if 0 <= v < 3 else "–"


def top_rows(model, l, u, pole, n=64) -> list[dict]:
    b = bundle(model)
    toks, vals = b["top_tok"][l, pole, u, :n].tolist(), b["top_val"][l, pole, u, :n].float().tolist()
    ss, bu, pl = (b[f"top_{k}"][l, pole, u, :n].tolist() for k in ("ss3", "burial", "plddt"))
    rows = []
    for k, (t, v) in enumerate(zip(toks, vals)):
        if t < 0:
            continue
        w = window(t)
        rows.append({"rank": k + 1, "act": round(v, 3), **w, "ss3": _ss3(ss[k]),
                     "burial": bu[k] if ss[k] >= 0 else None, "pLDDT": pl[k] if ss[k] >= 0 else None})
    return rows


def composition(toks) -> dict:
    res, prots = [], set()
    s = seqs()
    for t in toks:
        if t < 0:
            continue
        q, p = divmod(int(t), 1024)
        a = int(s["off"][q])
        res.append(chr(s["blob"][a + p])) if a + p < int(s["off"][q + 1]) else None
        prots.add(q)
    c = Counter(res)
    n = max(1, len(res))
    top = c.most_common(4)
    return {"residues": " ".join(f"{r}{k}" for r, k in top), "max_share": round(top[0][1] / n, 2) if top else 0.0,
            "proteins": len(prots), "n": len(res)}


def range_rows(model, l, u, pole, per_bin=6) -> list[dict]:
    b = bundle(model)
    m = b["meta"]
    mass = b["bin_mass"][l, pole, u].double()
    tot = float(m["n_sub"])
    rows = []
    for bi, q in enumerate(m["bin_quantiles"]):
        toks = b["res_tok"][l, pole, u, bi].tolist()
        vals = b["res_val"][l, pole, u, bi].float().tolist()
        samples = [f"{window(t)['context']} ({v:+.2f})" for t, v in zip(toks, vals) if t >= 0][:per_bin]
        comp = composition(toks)
        rows.append({"bin": f">= q{q}", "edge": round(float(b["edges"][l, pole, u, bi]), 3),
                     "share_of_tokens": f"{float(mass[bi]) / tot:.2e}", "residues": comp["residues"],
                     "max_share": comp["max_share"], "samples": " · ".join(samples)})
    return rows


def concentration_pct(model, l, u, pole) -> dict:
    """Max single-residue share of the top-K, as a percentile of the layer, trained and (if present) control."""
    b = bundle(model)

    def shares(key):
        T = b[key][l, pole]
        s = seqs()
        out = []
        for row in T.tolist():
            res = []
            for t in row:
                if t < 0:
                    continue
                q, p = divmod(int(t), 1024)
                res.append(int(s["blob"][int(s["off"][q]) + p]))
            out.append(Counter(res).most_common(1)[0][1] / max(1, len(res)) if res else 0.0)
        return np.array(out)
    key = ("conc", model, l, pole)
    if key not in _B:
        _B[key] = shares("top_tok")
        _B[key + ("ctrl",)] = shares("ctrl_top_tok") if "ctrl_top_tok" in b else None
    tr, ct = _B[key], _B[key + ("ctrl",)]
    x = tr[u]
    return {"max_share": round(float(x), 2), "pct_in_layer": round(100 * float((tr <= x).mean()), 1),
            "layer_median": round(float(np.median(tr)), 2),
            "control_median": None if ct is None else round(float(np.median(ct)), 2),
            "pct_vs_control": None if ct is None else round(100 * float((ct <= x).mean()), 1)}


def family_rows(model, l, u, pole) -> dict:
    b = bundle(model)
    out = {}
    for name, ik, vk in (("co-activation", "peer_coact_idx", "peer_coact_r"), ("input weights", "peer_input_idx", "peer_input_cos")):
        if ik not in b:
            continue
        rows = []
        for j, v in zip(b[ik][l, u].tolist(), b[vk][l, u].float().tolist()):
            c = composition(b["top_tok"][l, pole, j].tolist())
            rows.append({"unit": f"L{l} U{j}", "similarity": round(v, 3), "top residues": c["residues"],
                         "max_share": c["max_share"], "proteins": c["proteins"],
                         "helix_frac": profile(model, l, j, pole)["top"]["helix"]})
        out[name] = rows
    return out


def wiring(model, l, u) -> dict:
    b = bundle(model)
    aa = b["meta"]["aa_order"]
    lg = b["logit_aa"][l, u].float()
    lg = lg - lg.mean()
    order = torch.argsort(lg, descending=True).tolist()
    writes = {"promotes": ", ".join(f"{aa[i]} {lg[i]:+.3f}" for i in order[:4]),
              "suppresses": ", ".join(f"{aa[i]} {lg[i]:+.3f}" for i in order[::-1][:4])}
    readers = []
    for h in (0, 1):
        if l + h + 1 >= b["meta"]["L"]:
            continue
        for j, v in zip(b["reader_idx"][l, h, u].tolist(), b["reader_cos"][l, h, u].float().tolist()):
            if j < 0:
                continue
            c = composition(b["top_tok"][l + h + 1, 0, j].tolist())
            readers.append({"reader": f"L{l + h + 1} U{j}", "cos": round(v, 3), "reader top residues": c["residues"],
                            "max_share": c["max_share"]})
    return {"writes": writes, "readers": readers}


def profile(model, l, u, pole) -> dict:
    """Descriptor profile of the top-K and of each reservoir bin, against base rates."""
    b = bundle(model)

    def summ(ss, bu, pl):
        ss, bu, pl = ss.long(), bu.float(), pl.float()
        have = ss >= 0
        conf = have & (pl >= 70)
        f = lambda c: round(float((ss[conf] == c).float().mean()), 2) if conf.any() else None
        return {"n_desc": int(have.sum()), "loop": f(0), "helix": f(1), "strand": f(2),
                "burial": round(float(bu[have].mean()), 1) if have.any() else None,
                "pLDDT": round(float(pl[have].mean()), 1) if have.any() else None,
                "pLDDT<50": round(float((pl[have] < 50).float().mean()), 2) if have.any() else None}
    top = summ(b["top_ss3"][l, pole, u], b["top_burial"][l, pole, u], b["top_plddt"][l, pole, u])
    bins = []
    for bi, q in enumerate(b["meta"]["bin_quantiles"]):
        r = summ(b["res_ss3"][l, pole, u, bi], b["res_burial"][l, pole, u, bi], b["res_plddt"][l, pole, u, bi])
        bins.append({"bin": f">= q{q}", **r})
    base = b["base"]
    return {"top": top, "bins": bins,
            "base": {"loop": round(base["ss3_frac_conf"][0], 2), "helix": round(base["ss3_frac_conf"][1], 2),
                     "strand": round(base["ss3_frac_conf"][2], 2), "burial": round(base["burial_mean"], 1),
                     "burial_q10_50_90": [round(x, 1) for x in base["burial_q"]], "pLDDT": round(base["plddt_mean"], 1),
                     "pLDDT<50": round(base["plddt_lt50"], 2)}}


def enrichment_rows(model, l, u, pole) -> list[dict]:
    b = bundle(model)
    if "enr_ids" not in b:
        return []
    vocab = seqs()["vocab"]
    rows = []
    for li, lev in enumerate(b["meta"]["enrich_levels"]):
        above = int(b["enr_above"][l, pole, li, u])
        for i, c, r in zip(b["enr_ids"][l, pole, li, u].tolist(), b["enr_count"][l, pole, li, u].tolist(),
                           b["enr_ratio"][l, pole, li, u].float().tolist()):
            if i >= 0:
                rows.append({"level": f">= q{b['meta']['bin_quantiles'][lev]}", "annotation": vocab[i] if i < len(vocab) else i,
                             "count": c, "tokens_above": above, "enrichment": round(r, 1)})
    return rows


# ------------------------------------------------------------------ prediction, outcome, records
PRED_CHOICES = {"secondary structure": ["helix", "strand", "loop", "no preference"],
                "burial": ["buried", "exposed", "no preference"],
                "disorder": ["ordered", "disordered", "no preference"]}


def outcome(model, l, u, pole) -> dict:
    """Fixed rules (written before any reading) for what the top-K shows against base rates."""
    p = profile(model, l, u, pole)
    t, base = p["top"], p["base"]
    ss = "no preference"
    if t["helix"] is not None:
        diffs = {k: t[k] - base[k] for k in ("helix", "strand", "loop")}
        k, d = max(diffs.items(), key=lambda kv: kv[1])
        ss = k if d >= 0.15 else "no preference"
    q10, q50, q90 = base["burial_q10_50_90"]
    bur = "no preference" if t["burial"] is None else ("buried" if t["burial"] >= (q50 + q90) / 2 else "exposed" if t["burial"] <= (q10 + q50) / 2 else "no preference")
    dis = "no preference"
    if t["pLDDT<50"] is not None:
        dis = "disordered" if t["pLDDT<50"] >= base["pLDDT<50"] + 0.2 else "ordered" if (t["pLDDT<50"] <= max(0, base["pLDDT<50"] - 0.05) and t["pLDDT"] >= 85) else "no preference"
    return {"secondary structure": ss, "burial": bur, "disorder": dis, "profile": p}


def _write(path: Path, rec: dict):
    READINGS.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        json.dump(rec, f, indent=1)


def save_prediction(model, l, u, pole, pred: dict) -> dict:
    sid = uuid.uuid4().hex
    rec = {"session": sid, "unit": {"model": model, "layer": l, "unit": u, "pole": pole}, "bundle": digest(model),
           "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "prediction": pred}
    _write(READINGS / f"{sid}.prediction.json", rec)
    oc = outcome(model, l, u, pole)
    hits = {k: pred.get(k) == oc[k] for k in PRED_CHOICES}
    out = {"session": sid, "observed": {k: oc[k] for k in PRED_CHOICES}, "hits": hits}
    _write(READINGS / f"{sid}.outcome.json", out)
    return {**out, "profile": oc["profile"]}


def save_reading(sid: str, fields: dict) -> str:
    p = READINGS / f"{sid}.reading.json"
    _write(p, {"session": sid, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "fields": fields})
    return str(p)
