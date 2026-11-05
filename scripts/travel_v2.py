"""Travel, re-measured (predictions: results/mechanistic_round1_predictions.md section 2).

For the positive-extreme cases of measurement #3 (results/unit_writes_8M_20261105/cases.jsonl): silence the unit at
its position i and record the KL at every position (not only the largest elsewhere), by distance from i; the same
for a control unit at the same position matched on write magnitude |GELU(h)| x |w_out|; and a masked task: a
random 15% of the other positions masked, change in log p(true residue) there, next to the unmasked change at
the same positions.

  python scripts/travel_v2.py --cases results/unit_writes_8M_20261105/cases.jsonl --data swissprot.pt --out dir
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.esm import MODELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="8M")
    ap.add_argument("--level", default="positive extreme")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    from transformers import EsmForMaskedLM, EsmTokenizer
    hf = MODELS.get(a.model, a.model)
    tok = EsmTokenizer.from_pretrained(hf)
    model = EsmForMaskedLM.from_pretrained(hf).float().eval()
    layers = model.esm.encoder.layer
    d = torch.load(a.data, weights_only=False)
    idx = {acc: i for i, acc in enumerate(d["acc"])}
    cases = [json.loads(l) for l in open(a.cases) if l.strip()]
    cases = [c for c in cases if c["level"] == a.level]
    cap = {}
    for l, lay in enumerate(layers):
        lay.intermediate.dense.register_forward_hook(lambda _m, _i, o, l=l: cap.__setitem__(l, o[0]))
    edit = {"l": None, "u": None, "p": None}

    def mk(l):
        def hook(_m, _i, o):
            if edit["l"] == l:
                o = o.clone(); o[0, edit["p"] + 1, edit["u"]] = 0.0
            return o
        return hook
    for l, lay in enumerate(layers):
        lay.intermediate.register_forward_hook(mk(l))

    def lp(x, l=None, u=None, p=None):
        edit.update(l=l, u=u, p=p)
        r = torch.log_softmax(model(input_ids=x).logits[0].double(), -1)
        edit.update(l=None)
        return r

    def kl(p0, p1):
        return (p0.exp() * (p0 - p1)).sum(-1).clamp(min=0)
    g = torch.Generator().manual_seed(a.seed)
    rows = []
    for c in cases:
        s = d["seq"][idx[c["acc"]]][:1022]
        x = tok(s, return_tensors="pt")["input_ids"]; n = len(s); i, l, u = c["pos"], c["layer"], c["unit"]
        p0 = lp(x)
        gl = torch.nn.functional.gelu(cap[l][i + 1])
        wm = gl.abs() * layers[l].output.dense.weight.norm(dim=0)
        cand = [v for v in range(wm.numel()) if v != u]
        cu = min(cand, key=lambda v: abs(float(wm[v] - wm[u])))
        others = [q for q in range(n) if q != i]
        mpos = sorted(torch.tensor(others)[torch.randperm(len(others), generator=g)[: max(1, int(0.15 * n))]].tolist())
        xm = x.clone(); xm[0, [q + 1 for q in mpos]] = tok.mask_token_id
        tm = x[0, [q + 1 for q in mpos]]
        pm0 = lp(xm)
        r = {"layer": l, "unit": u, "acc": c["acc"], "pos": i, "write": float(wm[u]), "ctl_unit": cu, "ctl_write": float(wm[cu])}
        for tag, uu in (("unit", u), ("ctl", cu)):
            p1 = lp(x, l, uu, i)
            k = kl(p0[1:-1], p1[1:-1])                                      # per residue position
            dist = (torch.arange(n) - i).abs()
            o = dist > 0
            near, mid, far = (dist >= 1) & (dist <= 3), (dist >= 4) & (dist <= 16), dist > 16
            md = lambda m: float(k[m].median()) if int(m.sum()) else None  # noqa: E731
            pm1 = lp(xm, l, uu, i)
            mi = torch.tensor([q + 1 for q in mpos])
            dmask = (pm1[mi, tm] - pm0[mi, tm]).abs()
            dunm = (p1[mi, tm] - p0[mi, tm]).abs()
            r[tag] = {"kl_at": float(k[i]), "kl_else_median": md(o), "kl_else_sum": float(k[o].sum()), "kl_else_max": float(k[o].max()) if int(o.sum()) else 0.0,
                      "kl_near_median": md(near), "kl_mid_median": md(mid), "kl_far_median": md(far),
                      "masked_median_abs": float(dmask.median()), "unmasked_same_positions_median_abs": float(dunm.median()),
                      "masked_sum_abs": float(dmask.sum())}
        rows.append(r)
    (out / "cases.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def med(v):
        v = [x for x in v if x is not None]
        return round(st.median(v), 8) if v else None
    summ = {}
    for l in sorted({r["layer"] for r in rows}):
        R = [r for r in rows if r["layer"] == l]
        f = lambda key, tag="unit": [r[tag][key] for r in R]  # noqa: E731
        ratio = lambda num, den, tag="unit": [r[tag][num] / r[tag][den] for r in R if r[tag][den] and r[tag][num] is not None]  # noqa: E731
        summ[str(l)] = {"n": len(R), "kl_at": med(f("kl_at")), "kl_else_median": med(f("kl_else_median")),
                        "else_median_over_at": med(ratio("kl_else_median", "kl_at")), "else_sum_over_at": med(ratio("kl_else_sum", "kl_at")),
                        "near_over_far": med([r["unit"]["kl_near_median"] / r["unit"]["kl_far_median"] for r in R
                                              if r["unit"]["kl_far_median"] and r["unit"]["kl_near_median"] is not None]),
                        "unit_over_ctl_else_sum": med([r["unit"]["kl_else_sum"] / r["ctl"]["kl_else_sum"] for r in R if r["ctl"]["kl_else_sum"] > 0]),
                        "unit_over_ctl_kl_at": med([r["unit"]["kl_at"] / r["ctl"]["kl_at"] for r in R if r["ctl"]["kl_at"] > 0]),
                        "write_unit_over_ctl": med([r["write"] / r["ctl_write"] for r in R if r["ctl_write"] > 0]),
                        "masked_median_abs": med(f("masked_median_abs")), "masked_over_unmasked": med(ratio("masked_median_abs", "unmasked_same_positions_median_abs")),
                        "ctl_masked_median_abs": med(f("masked_median_abs", "ctl"))}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
