"""Mechanistic case: two units of one layer that read similar inputs and write opposed vectors (default: ESM-2 8M
layer 1, units 86 and 1200). Predictions: results/mechanistic_round1_predictions.md section 1.

On fresh proteins (the date-held-out set): positions where both units fire (both above their q99.9 there) and,
where they exist, where only one does (one above q99.9, the other below q90); plus "both_q99_only" (both above
q99 but not both above q99.9), the originally predicted definition, where the units barely write. At each position, one at a time
(post-GELU, that position only): silence a; silence b; silence both; halve both. Controls per position: for
each of a and b, the other unit of the layer whose write magnitude |GELU(h)| x |w_out| there is closest; the two
form the control pair and get the same three silencings. Readouts: log-probabilities at the position
(unmasked), pre-GELU activations of every later layer at the position, and the change in log p(true residue) at
a random 15% of the other positions when they are masked.

  python scripts/pair_silencing.py --data swissprot.pt --held lists/heldout.tsv --out dir [--layer 1 --a 86 --b 1200]
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

AA = "ACDEFGHIKLMNPQRSTVWY"


def spearman(x, y):
    rx = torch.argsort(torch.argsort(x)).double(); ry = torch.argsort(torch.argsort(y)).double()
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / torch.sqrt((rx * rx).sum() * (ry * ry).sum()))


def cos(x, y):
    return float(torch.nn.functional.cosine_similarity(x.flatten().double(), y.flatten().double(), dim=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8M")
    ap.add_argument("--data", required=True)
    ap.add_argument("--held", required=True)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--a", type=int, default=86)
    ap.add_argument("--b", type=int, default=1200)
    ap.add_argument("--proteins", type=int, default=400)
    ap.add_argument("--max-both", type=int, default=60)
    ap.add_argument("--max-only", type=int, default=30)
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
    L, D, l0 = len(layers), model.config.intermediate_size, a.layer
    Wout_norm = layers[l0].output.dense.weight.norm(dim=0)
    aa_ids = torch.tensor([tok.convert_tokens_to_ids(c) for c in AA])

    d = torch.load(a.data, weights_only=False)
    idx = {acc: i for i, acc in enumerate(d["acc"])}
    held = [ln.split("\t")[0].split(",")[0].strip() for ln in open(a.held) if ln.strip() and not ln.startswith("Entry")]
    held = [x for x in held if x in idx]
    g = torch.Generator().manual_seed(a.seed)
    pick = [held[i] for i in torch.randperm(len(held), generator=g)[: a.proteins].tolist()]
    seqs = [d["seq"][idx[x]][:1022] for x in pick]

    cap = {}
    hk = [layers[l].intermediate.dense.register_forward_hook(lambda _m, _i, o, l=l: cap.__setitem__(l, o[0])) for l in range(L)]
    # pass 1: the two units on every token of the fresh proteins
    ha, hb, res, where = [], [], [], []
    for j, s in enumerate(seqs):
        x = tok(s, return_tensors="pt")["input_ids"]
        model(input_ids=x)
        H = cap[l0][1:-1]
        ha.append(H[:, a.a].clone()); hb.append(H[:, a.b].clone())
        res += list(s); where += [(j, p) for p in range(len(s))]
    ha, hb = torch.cat(ha), torch.cat(hb)
    qa99, qb99, qa90, qb90 = (float(torch.quantile(v, q)) for v, q in ((ha, 0.999), (hb, 0.999), (ha, 0.90), (hb, 0.90)))
    qa_lo, qb_lo = float(torch.quantile(ha, 0.99)), float(torch.quantile(hb, 0.99))   # the original (q99) definition, kept as a group
    top = lambda v: [res[i] for i in torch.topk(v, 64).indices.tolist()]  # noqa: E731
    pre = {"tokens": int(ha.numel()), "pearson": float(torch.corrcoef(torch.stack([ha, hb]))[0, 1]),
           "top64_V_a": top(ha).count("V") / 64, "top64_V_b": top(hb).count("V") / 64,
           "q99.9 (selection)": [qa99, qb99], "q99": [qa_lo, qb_lo], "q90": [qa90, qb90]}

    def choose(mask, m):
        ix = torch.nonzero(mask).flatten().tolist()
        seen, keep = set(), []
        for i in [ix[k] for k in torch.randperm(len(ix), generator=g).tolist()]:
            if where[i][0] not in seen:
                seen.add(where[i][0]); keep.append(where[i])
            if len(keep) >= m:
                break
        return keep
    groups = {"both": choose((ha >= qa99) & (hb >= qb99), a.max_both),
              "both_q99_only": choose((ha >= qa_lo) & (hb >= qb_lo) & ~((ha >= qa99) & (hb >= qb99)), a.max_only),
              "a_only": choose((ha >= qa99) & (hb < qb90), a.max_only),
              "b_only": choose((hb >= qb99) & (ha < qa90), a.max_only)}
    pre["positions"] = {k: len(v) for k, v in groups.items()}
    pre["available"] = {"both": int(((ha >= qa99) & (hb >= qb99)).sum()), "a_only": int(((ha >= qa99) & (hb < qb90)).sum()),
                        "b_only": int(((hb >= qb99) & (ha < qa90)).sum())}
    print(json.dumps(pre), flush=True)

    edit = {"units": [], "scale": 0.0, "pos": None}

    def hook(_m, _i, o):
        if edit["units"]:
            o = o.clone(); o[0, edit["pos"] + 1, edit["units"]] *= edit["scale"]
        return o
    layers[l0].intermediate.register_forward_hook(hook)

    def run(x, units, scale, p):
        edit.update(units=units, scale=scale, pos=p)
        lp = torch.log_softmax(model(input_ids=x).logits[0].double(), -1)
        acts = torch.stack([cap[l][p + 1].clone() for l in range(l0 + 1, L)])  # later layers, pre-GELU, at p
        edit.update(units=[])
        return lp, acts

    rows = []
    for gname, poss in groups.items():
        for (j, p) in poss:
            s = seqs[j]
            x = tok(s, return_tensors="pt")["input_ids"]
            n = len(s)
            others = [q for q in range(n) if q != p]
            mpos = sorted(torch.tensor(others)[torch.randperm(len(others), generator=g)[: max(1, int(0.15 * n))]].tolist())
            xm = x.clone(); xm[0, [q + 1 for q in mpos]] = tok.mask_token_id
            true_m = x[0, [q + 1 for q in mpos]]
            lp0, ac0 = run(x, [], 1.0, p)
            gl = torch.nn.functional.gelu(cap[l0][p + 1])                     # layer-l0 post-GELU at p (unedited)
            wm = gl.abs() * Wout_norm
            cand = [u for u in range(D) if u not in (a.a, a.b)]
            ca = min(cand, key=lambda u: (abs(float(wm[u] - wm[a.a])), abs(float(gl[u] - gl[a.a]))))
            cb = min([u for u in cand if u != ca], key=lambda u: (abs(float(wm[u] - wm[a.b])), abs(float(gl[u] - gl[a.b]))))
            lpm0, _ = run(xm, [], 1.0, p)
            conds = {"sil_a": ([a.a], 0.0), "sil_b": ([a.b], 0.0), "sil_ab": ([a.a, a.b], 0.0), "half_ab": ([a.a, a.b], 0.5),
                     "ctl_a": ([ca], 0.0), "ctl_b": ([cb], 0.0), "ctl_ab": ([ca, cb], 0.0)}
            r = {"group": gname, "protein": pick[j], "pos": p, "res": s[p], "g_a": float(gl[a.a]), "g_b": float(gl[a.b]),
                 "ctl": [ca, cb], "g_ctl": [float(gl[ca]), float(gl[cb])], "write": [float(wm[a.a]), float(wm[a.b]), float(wm[ca]), float(wm[cb])]}
            D_l, D_act = {}, {}
            for c, (units, sc) in conds.items():
                lp, ac = run(x, units, sc, p)
                lpm, _ = run(xm, units, sc, p)
                D_l[c] = (lp[p + 1, aa_ids] - lp0[p + 1, aa_ids])                       # 20 residues, at p
                D_act[c] = ac - ac0
                kl = float((lp0[p + 1].exp() * (lp0[p + 1] - lp[p + 1])).sum())
                dm = (lpm[[q + 1 for q in mpos], true_m] - lpm0[[q + 1 for q in mpos], true_m]).abs()
                r[c] = {"kl": max(kl, 0.0), "dlogp_norm": float(D_l[c].norm()), "masked_mean_abs": float(dm.mean()),
                        "dlogp_by_residue": {AA[k]: round(float(D_l[c][k]), 5) for k in range(20)}}
            for pre_, (x1, x2, xb) in {"pair": ("sil_a", "sil_b", "sil_ab"), "ctl": ("ctl_a", "ctl_b", "ctl_ab")}.items():
                denom = float(D_l[x1].norm() + D_l[x2].norm())
                r[pre_ + "_cos_dlogp"] = cos(D_l[x1], D_l[x2])
                r[pre_ + "_subadditivity"] = float(D_l[xb].norm()) / denom if denom > 0 else None
                r[pre_ + "_spearman_next_layer"] = spearman(D_act[x1][0], D_act[x2][0])
            r["half_over_both"] = float(D_l["half_ab"].norm() / D_l["sil_ab"].norm()) if float(D_l["sil_ab"].norm()) > 0 else None
            r["kl_both_lt_kl_a"] = r["sil_ab"]["kl"] < r["sil_a"]["kl"]
            r["top_next_units_sil_a"] = torch.topk(D_act["sil_a"][0].abs(), 10).indices.tolist()
            rows.append(r)
    (out / "cases.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def med(gr, key, sub=None):
        v = [(r[key][sub] if sub else r[key]) for r in rows if r["group"] == gr and (r[key] if sub is None else r[key][sub]) is not None]
        return round(st.median(v), 4) if v else None
    summ = {"pre": pre}
    for gr in groups:
        R = [r for r in rows if r["group"] == gr]
        if not R:
            continue
        resid = {k: round(st.mean(r["sil_a"]["dlogp_by_residue"][k] for r in R), 4) for k in AA}
        residb = {k: round(st.mean(r["sil_b"]["dlogp_by_residue"][k] for r in R), 4) for k in AA}
        from collections import Counter
        nxt = Counter(u for r in R for u in r["top_next_units_sil_a"]).most_common(8)
        summ[gr] = {"n": len(R), **{f"{k}": med(gr, k) for k in ("pair_cos_dlogp", "ctl_cos_dlogp", "pair_subadditivity", "ctl_subadditivity",
                                                                    "pair_spearman_next_layer", "ctl_spearman_next_layer", "half_over_both")},
                    "frac_kl_both_lt_kl_a": round(sum(r["kl_both_lt_kl_a"] for r in R) / len(R), 3),
                    **{f"kl_{c}": med(gr, c, "kl") for c in ("sil_a", "sil_b", "sil_ab", "half_ab", "ctl_a", "ctl_b", "ctl_ab")},
                    **{f"masked_{c}": med(gr, c, "masked_mean_abs") for c in ("sil_a", "sil_b", "sil_ab", "ctl_a", "ctl_b", "ctl_ab")},
                    "mean_dlogp_by_residue_sil_a": dict(sorted(resid.items(), key=lambda kv: kv[1])[:3] + sorted(resid.items(), key=lambda kv: kv[1])[-3:]),
                    "mean_dlogp_by_residue_sil_b": dict(sorted(residb.items(), key=lambda kv: kv[1])[:3] + sorted(residb.items(), key=lambda kv: kv[1])[-3:]),
                    "next_layer_units_most_changed_by_sil_a": nxt,
                    "write_a_b_ctl_median": [round(st.median(r["write"][k] for r in R), 3) for k in range(4)]}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
