"""Real-unit cases at 8M (results/cases_round2_plan.md section 1).

  select   : apply the frozen selection rule to calls_pre.json (seed 20261105, eligibility n_clusters >= 40)
  discover : per unit, on discovery material only: atlas examples (top-64, peak and weaker reservoir bins), ruler
             scores, strongest next-layer readers (cosine of this unit's write with each next-layer input weight),
             and silencing on discovery proteins at positions above the unit's discovery q99.9: signed change in
             log p of each residue at the position when it is masked, and at the masked +-3 neighbours, against a
             control unit matched on write magnitude in the same masked forward
  confirm  : per unit, frozen predictions (JSON) tested on fresh proteins: (a) the predicted context at fresh top
             positions vs base; (b) silencing: signed change of the predicted readout residues' log p, and of
             log p(true), at the masked position, against the write-matched control

  python scripts/case_units.py select --calls calls_pre.json --atlas a8_pre.slim.pt --out sel.json
  python scripts/case_units.py discover --unit 3,517 --data .. --desc-dir .. --uniref .. --atlas .. --calls .. --out dir
  python scripts/case_units.py confirm --unit 3,517 --pred pred.json --data .. --desc-dir .. --uniref .. --held .. --exclude-json .. --out dir
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.core import SEQ_STRIDE  # noqa: E402
from scripts.pair_balance import burial3_of, proteins  # noqa: E402
from scripts.synthetic_columns import state5  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
CATS = {"structural association": 2, "composition association": 1, "context association": 1, "no association found by this procedure": 1}


def cmd_select(a):
    C = json.load(open(a.calls))
    at = torch.load(a.atlas, map_location="cpu", weights_only=False)
    elig = [c for c in C if c["levels"]["peak"]["n_clusters"] >= 40]
    cols = sorted((c["layer"], c["unit"]) for c in elig)
    perm = torch.randperm(len(cols), generator=torch.Generator().manual_seed(20261105)).tolist()
    by = {(c["layer"], c["unit"]): c for c in elig}
    out = {"eligible": len(elig), "random": {}, "deliberate": {}}
    deliberate = {(1, 86), (1, 1200)}
    for cat, k in CATS.items():
        picks = [cols[i] for i in perm if by[cols[i]]["call"] == cat and cols[i] not in deliberate][:k + 3]   # k + 3 fallbacks in order
        out["random"][cat] = {"picks": picks[:k], "fallbacks_in_order": picks[k:]}
    # deliberate: "not detected" with the strongest single next-layer reader
    from transformers import EsmModel
    m = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D", add_pooling_layer=False)
    best = None
    for (l, u) in cols:
        if by[(l, u)]["call"] != "no association found by this procedure" or l >= 5:
            continue
        w = m.encoder.layer[l].output.dense.weight[:, u].detach()
        Wn = m.encoder.layer[l + 1].intermediate.dense.weight.detach()
        c = float(torch.nn.functional.cosine_similarity(Wn, w[None, :], dim=1).max())
        if best is None or c > best[0] + 1e-12:
            best = (c, l, u)
    out["deliberate"]["not detected, strongest single reader"] = {"unit": [best[1], best[2]], "max_reader_cosine": best[0]}
    out["counts_by_call"] = dict(Counter(c["call"] for c in C))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


class Model:
    def __init__(self):
        from transformers import EsmForMaskedLM, EsmTokenizer
        hf = "facebook/esm2_t6_8M_UR50D"
        self.tok = EsmTokenizer.from_pretrained(hf); self.m = EsmForMaskedLM.from_pretrained(hf).float().eval()
        self.layers = self.m.esm.encoder.layer
        self.cap = {}; self.edit = None
        for l, lay in enumerate(self.layers):
            lay.intermediate.dense.register_forward_hook(lambda _m, _i, o, l=l: self.cap.__setitem__(("h", l), o[0]))
            lay.intermediate.register_forward_hook(self._mk(l))
        self.aa_ids = torch.tensor([self.tok.convert_tokens_to_ids(c) for c in AA])

    def _mk(self, l):
        def hook(_m, _i, o):
            self.cap[("g", l)] = o[0]
            if self.edit and self.edit[0] == l:
                o = o.clone(); o[0, self.edit[1] + 1, self.edit[2]] = 0.0
            return o
        return hook

    def run(self, s_or_ids, edit=None):
        ids = s_or_ids if torch.is_tensor(s_or_ids) else self.tok(s_or_ids, return_tensors="pt")["input_ids"]
        self.edit = edit
        with torch.no_grad():
            lp = torch.log_softmax(self.m(input_ids=ids).logits[0].double(), -1)[:, self.aa_ids]
        self.edit = None
        return lp


def unit_positions(M, prot, l, u, thr, need_margin=3, cap_per_protein=1):
    out = []
    for j, p in enumerate(prot):
        M.run(p["seq"])
        h = M.cap[("h", l)][1:-1, u]
        idx = torch.nonzero(h >= thr).flatten().tolist()
        idx = [i for i in idx if need_margin <= i < len(p["seq"]) - need_margin][:cap_per_protein]
        out += [(j, i, float(h[i])) for i in idx]
    return out


def silence(M, p, i, l, u):
    """Mask position i; silence u at i (post-GELU 0) and a write-matched control in the same masked forward."""
    ids = M.tok(p["seq"], return_tensors="pt")["input_ids"]
    x = ids.clone(); x[0, i + 1] = M.tok.mask_token_id
    lp0 = M.run(x)
    g = M.cap[("g", l)][i + 1]
    wm = g.abs() * M.layers[l].output.dense.weight.norm(dim=0)
    order = torch.argsort((wm - wm[u]).abs()).tolist()
    ctl = next(v for v in order if v != u)
    d_u = (M.run(x, (l, i, u))[i + 1] - lp0[i + 1])
    d_c = (M.run(x, (l, i, ctl))[i + 1] - lp0[i + 1])
    true = AA.find(p["seq"][i])
    return d_u, d_c, true, float(wm[u]), float(wm[ctl])


def cmd_discover(a):
    l, u = map(int, a.unit.split(","))
    at = torch.load(a.atlas, map_location="cpu", weights_only=False)
    d = torch.load(a.data, weights_only=False)
    calls = {(c["layer"], c["unit"]): c for c in json.load(open(a.calls))}
    c = calls[(l, u)]

    def comp(toks):
        r = Counter(); pr = set()
        for t in toks:
            if t < 0:
                continue
            si, pos = divmod(int(t), SEQ_STRIDE); s = d["seq"][si]
            if pos < len(s):
                r[s[pos]] += 1; pr.add(si)
        return {"residues": r.most_common(6), "proteins": len(pr)}
    ex = {"top64": comp(at["top_tok"][l, 0, u].tolist()), "top64_values": [round(float(at["top_val"][l, 0, u, 0]), 2), round(float(at["top_val"][l, 0, u, -1]), 2)],
          "edges": [round(float(v), 2) for v in at["edges"][l, 0, u]]}
    for b, name in ((5, "peak (>= q99.999)"), (3, "q99.9-q99.99"), (2, "q99-q99.9"), (1, "q90-q99")):
        ex[name] = comp(at["res_tok"][l, 0, u, b].tolist())
    M = Model()
    readers = None
    if l < 5:
        w = M.layers[l].output.dense.weight[:, u].detach()
        cs = torch.nn.functional.cosine_similarity(M.layers[l + 1].intermediate.dense.weight.detach(), w[None, :], dim=1)
        top = torch.topk(cs, 5)
        readers = [(int(v), round(float(x), 3)) for x, v in zip(top.values, top.indices)]
        neg = torch.topk(-cs, 3)
        readers_neg = [(int(v), round(-float(x), 3)) for x, v in zip(neg.values, neg.indices)]
    prot = proteins(a, False, a.n, a.seed)
    b3, _ = burial3_of(prot)
    allh = []
    for p in prot:
        M.run(p["seq"]); allh.append(M.cap[("h", l)][1:-1, u].clone())
    q999 = float(torch.quantile(torch.cat(allh), 0.999))
    pos = unit_positions(M, prot, l, u, q999)
    ctx = Counter(); s5 = Counter(); bb = Counter()
    du, dc = [], []
    tr_u, tr_c = [], []
    for (j, i, hv) in pos[: a.max_positions]:
        p = prot[j]
        ctx[p["seq"][i]] += 1; s5[int(state5(p)[i])] += 1; bb[int(b3[j][i])] += 1
        d_u, d_c, true, wu, wc = silence(M, p, i, l, u)
        du.append(d_u); dc.append(d_c)
        if true >= 0:
            tr_u.append(float(d_u[true])); tr_c.append(float(d_c[true]))
    DU = torch.stack(du).mean(0) if du else torch.zeros(20); DC = torch.stack(dc).mean(0) if dc else torch.zeros(20)
    order = torch.argsort(DU)
    res = {"unit": [l, u], "call": {k: c.get(k) for k in ("call", "primary", "modes", "residue_modes", "aa_top", "persistence", "mode_label")},
           "levels": c["levels"], "examples": ex, "readers_next_layer": readers, "readers_next_layer_negative": readers_neg if l < 5 else None,
           "discovery": {"proteins": len(prot), "q99.9": q999, "positions": len(du),
                         "centre_residues": ctx.most_common(6), "state5": dict(s5), "burial3": dict(bb),
                         "silence_mean_dlogp_by_residue": {AA[k]: round(float(DU[k]), 4) for k in order.tolist()},
                         "control_mean_abs_dlogp": round(float(DC.abs().mean()), 5), "unit_mean_abs_dlogp": round(float(DU.abs().mean()), 5),
                         "dlogp_true_median_unit_ctl": [round(st.median(tr_u), 5) if tr_u else None, round(st.median(tr_c), 5) if tr_c else None]}}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"discover_{l}_{u}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def cmd_confirm(a):
    l, u = map(int, a.unit.split(","))
    P = json.load(open(a.pred))
    first = json.load(open(a.exclude_json)) if a.exclude_json else []
    M = Model()
    prot = proteins(a, True, a.n, a.seed, exclude=first)
    b3, _ = burial3_of(prot)
    thr = P["q99.9_discovery"]
    pos = unit_positions(M, prot, l, u, thr)
    feat = P["context"]                                   # {"kind": "residue"|"state5"|"burial3", "values": [...]}
    def has(j, i):
        p = prot[j]
        if feat["kind"] == "residue":
            return p["seq"][i] in feat["values"]
        if feat["kind"] == "state5":
            return int(state5(p)[i]) in feat["values"]
        return int(b3[j][i]) in feat["values"]
    base_n = base_hit = 0
    for j, p in enumerate(prot):
        for i in range(0, len(p["seq"]), 7):
            base_n += 1; base_hit += has(j, i)
    hit = sum(has(j, i) for (j, i, _) in pos)
    raise_set = [AA.index(c) for c in P["readout_residues"]]      # residues whose log p the unit is predicted to raise
    ru, rc, tu, tc = [], [], [], []
    for (j, i, hv) in pos[: a.max_positions]:
        d_u, d_c, true, wu, wc = silence(M, prot[j], i, l, u)
        ru.append(float(d_u[raise_set].mean())); rc.append(float(d_c[raise_set].mean()))
        if true >= 0:
            tu.append(float(d_u[true])); tc.append(float(d_c[true]))
    def mci(v, B=2000):
        if not v:
            return None
        t = torch.tensor(v, dtype=torch.float64); g = torch.Generator().manual_seed(0)
        bs = torch.stack([t[torch.randint(len(t), (len(t),), generator=g)].mean() for _ in range(B)])
        return [round(float(t.mean()), 5), round(float(torch.quantile(bs, 0.025)), 5), round(float(torch.quantile(bs, 0.975)), 5)]
    res = {"unit": [l, u], "fresh_proteins": len(prot), "positions": len(pos), "context": feat,
           "context_share_at_top": round(hit / max(len(pos), 1), 3), "context_share_base": round(base_hit / max(base_n, 1), 3),
           "silence_readout_residues_mean_dlogp (mean, 95% CI)": {"unit": mci(ru), "control": mci(rc)},
           "silence_dlogp_true (mean, 95% CI)": {"unit": mci(tu), "control": mci(tc)}}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"confirm_{l}_{u}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("select"); s.add_argument("--calls", required=True); s.add_argument("--atlas", required=True); s.add_argument("--out", required=True)
    for name in ("discover", "confirm"):
        q = sp.add_parser(name)
        q.add_argument("--unit", required=True); q.add_argument("--data", required=True); q.add_argument("--desc-dir", required=True)
        q.add_argument("--uniref", required=True); q.add_argument("--out", required=True); q.add_argument("--n", type=int, default=600)
        q.add_argument("--max-positions", type=int, default=150)
        if name == "discover":
            q.add_argument("--atlas", required=True); q.add_argument("--calls", required=True); q.add_argument("--seed", type=int, default=2)
        else:
            q.add_argument("--pred", required=True); q.add_argument("--exclude-json", default=None); q.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    {"select": cmd_select, "discover": cmd_discover, "confirm": cmd_confirm}[a.cmd](a)


if __name__ == "__main__":
    main()
