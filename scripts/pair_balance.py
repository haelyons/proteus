"""The pair's balance (results/cases_round2_plan.md section 3). Two stages:

  discover: on discovery proteins (non-held-out, with descriptors) fit h_b ~ alpha + beta * h_a where either unit
            is above its q99; residual = departure from the fitted balance; contexts of departures (residues at
            -3..+3, state5, burial3); graded interventions (a and b scaled over {0, .5, 1, 1.5, 2}) at positions
            where both fire, read out at the masked +-3 neighbours: polar-minus-hydrophobic index (mean log p over
            E, Q, K, G minus over I, L, F), signed change in log p(true), and joint vs vector-sum of single effects.
  confirm : frozen parameters from a JSON (fit, thresholds, predicted context); fresh proteins (held-out, not the
            first experiment's 400); (a) do departures recur in the predicted context; (b) remove the natural
            variation (set b to the fitted value) at departure positions, signed change in log p(true) at masked
            +-3 neighbours, predicted context vs elsewhere, against controls matched on the perturbation norm
            applied in that masked forward.

  python scripts/pair_balance.py discover --data swissprot.pt --desc-dir desc_subset --uniref .. --out dir
  python scripts/pair_balance.py confirm --frozen frozen.json --data .. --desc-dir .. --uniref .. --held .. --out dir
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
from atlas.esm import MODELS  # noqa: E402
from scripts.synthetic_columns import load_subset, state5  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
POLAR, HYDRO = "EQKG", "ILF"


class Pair:
    def __init__(self, layer=1, a=86, b=1200, model="8M"):
        from transformers import EsmForMaskedLM, EsmTokenizer
        hf = MODELS[model]
        self.tok = EsmTokenizer.from_pretrained(hf)
        self.m = EsmForMaskedLM.from_pretrained(hf).float().eval()
        self.l, self.a, self.b = layer, a, b
        lay = self.m.esm.encoder.layer[layer]
        self.wout = lay.output.dense.weight.detach()                       # [d_model, d_ff]
        self.cap, self.edit = {}, None
        lay.intermediate.dense.register_forward_hook(lambda _m, _i, o: self.cap.__setitem__("h", o[0]))
        lay.intermediate.register_forward_hook(self._hook)
        self.aa_ids = torch.tensor([self.tok.convert_tokens_to_ids(c) for c in AA])
        self.pol = torch.tensor([AA.index(c) for c in POLAR]); self.hyd = torch.tensor([AA.index(c) for c in HYDRO])

    def _hook(self, _m, _i, o):
        if self.edit:
            o = o.clone()
            for (p, u, mode, v) in self.edit:
                if mode == "scale":
                    o[0, p + 1, u] *= v
                elif mode == "set":
                    o[0, p + 1, u] = v
                elif mode == "add":
                    o[0, p + 1, u] += v
            self.cap["post"] = o[0]
        else:
            self.cap["post"] = o[0]
        return o

    def run(self, ids, edit=None):
        self.edit = edit
        with torch.no_grad():
            lp = torch.log_softmax(self.m(input_ids=ids).logits[0].double(), -1)
        self.edit = None
        return lp[:, self.aa_ids]                                          # [L+2, 20]

    def ids(self, s):
        return self.tok(s, return_tensors="pt")["input_ids"]

    def masked(self, ids, positions):
        x = ids.clone(); x[0, [q + 1 for q in positions]] = self.tok.mask_token_id
        return x


def readouts(lp, lp0, positions, seq):
    """At the given (masked) residue positions: PH index change and signed change of log p(true)."""
    r = torch.tensor([q + 1 for q in positions])
    d = lp[r] - lp0[r]                                                     # [n, 20]
    true = torch.tensor([AA.index(seq[q]) if seq[q] in AA else 0 for q in positions])
    ok = torch.tensor([seq[q] in AA for q in positions])
    ph = lambda x: x[:, PAIR.pol].mean(1) - x[:, PAIR.hyd].mean(1)  # noqa: E731
    return {"d_ph": float((ph(lp[r]) - ph(lp0[r])).mean()), "d_true": float(d[torch.arange(len(positions)), true][ok].mean()) if ok.any() else 0.0,
            "vec": d.flatten()}


PAIR = None


def proteins(a, heldout, n, seed, exclude=()):
    data = torch.load(a.data, weights_only=False)
    prot = load_subset(a.desc_dir, data, a.uniref, heldout=heldout)
    prot = [p for p in prot if p["acc"] not in set(exclude)]
    g = torch.Generator().manual_seed(seed)
    return [prot[i] for i in torch.randperm(len(prot), generator=g)[:n].tolist()]


def burial3_of(prot):
    allb = torch.cat([p["burial"][p["plddt"] >= 70] for p in prot])
    cut = torch.quantile(allb, torch.tensor([1 / 3, 2 / 3]))
    out = []
    for p in prot:
        b = torch.full((len(p["seq"]),), -1, dtype=torch.long); ok = p["plddt"] >= 70
        b[ok] = torch.bucketize(p["burial"][ok], cut, right=True); out.append(b)
    return out, cut


def activations(prot):
    H = []
    for p in prot:
        PAIR.run(PAIR.ids(p["seq"]))
        H.append(PAIR.cap["h"][1:-1, [PAIR.a, PAIR.b]].clone())
    return H


def contexts(prot, b3, H, sel):
    """Residue log2 enrichment at -3..+3, state5 and burial3, for tokens in `sel` vs all candidate tokens."""
    def tally(mask_fn):
        c = Counter(); s5 = Counter(); bb = Counter(); n = 0
        for p, b, h in zip(prot, b3, H):
            m = mask_fn(h)
            for i in torch.nonzero(m).flatten().tolist():
                n += 1
                for o in range(-3, 4):
                    j = i + o
                    if 0 <= j < len(p["seq"]):
                        c[(o, p["seq"][j])] += 1
                s5[int(state5(p)[i])] += 1; bb[int(b[i])] += 1
        return c, s5, bb, n
    return tally


def cmd_discover(a):
    global PAIR
    PAIR = Pair()
    prot = proteins(a, False, a.n, a.seed)
    b3, cut = burial3_of(prot)
    H = activations(prot)
    allh = torch.cat(H)
    qa, qb = float(torch.quantile(allh[:, 0], 0.99)), float(torch.quantile(allh[:, 1], 0.99))
    qa9, qb9 = float(torch.quantile(allh[:, 0], 0.999)), float(torch.quantile(allh[:, 1], 0.999))
    M = (allh[:, 0] >= qa) | (allh[:, 1] >= qb)
    x, y = allh[M, 0].double(), allh[M, 1].double()
    beta = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()); alpha = float(y.mean() - beta * x.mean())
    res = y - (alpha + beta * x)
    lo, hi = float(torch.quantile(res, 0.10)), float(torch.quantile(res, 0.90))
    r2 = 1 - float((res ** 2).sum() / ((y - y.mean()) ** 2).sum())
    cand = lambda h: (h[:, 0] >= qa) | (h[:, 1] >= qb)  # noqa: E731
    rr = lambda h: h[:, 1] - (alpha + beta * h[:, 0])  # noqa: E731
    tally = contexts(prot, b3, H, None)
    base = tally(cand)
    heavy_b = tally(lambda h: cand(h) & (rr(h) >= hi))
    heavy_a = tally(lambda h: cand(h) & (rr(h) <= lo))

    def enrich(g):
        c, s5, bb, n = g; cb, s5b, bbb, nb = base
        res_ = sorted(((round(float(torch.log2(torch.tensor((c[k] + 1) / (n + 20)) / ((cb[k] + 1) / (nb + 20)))), 2), k[0], k[1], c[k])
                       for k in cb if cb[k] >= 30), reverse=True)
        return {"n": n, "top_enriched": res_[:12], "top_depleted": res_[-8:],
                "state5": {s: round(s5[s] / max(n, 1), 3) for s in range(5)}, "state5_base": {s: round(s5b[s] / max(nb, 1), 3) for s in range(5)},
                "burial3": {s: round(bb[s] / max(n, 1), 3) for s in (-1, 0, 1, 2)}, "burial3_base": {s: round(bbb[s] / max(nb, 1), 3) for s in (-1, 0, 1, 2)}}
    disc = {"proteins": len(prot), "tokens": int(allh.shape[0]), "q99": [qa, qb], "q99.9": [qa9, qb9], "fit": {"alpha": alpha, "beta": beta, "r2": r2,
            "n": int(M.sum())}, "residual_q10_q90": [lo, hi], "burial_cut": cut.tolist(),
            "pearson_all": float(torch.corrcoef(allh.T)[0, 1]),
            "contexts": {"b_heavy (residual >= q90)": enrich(heavy_b), "a_heavy (residual <= q10)": enrich(heavy_a)}}
    # graded interventions at positions where both fire (discovery), read out at masked +-3 neighbours
    pos = []
    for j, (p, h) in enumerate(zip(prot, H)):
        for i in torch.nonzero((h[:, 0] >= qa9) & (h[:, 1] >= qb9)).flatten().tolist():
            if 3 <= i < len(p["seq"]) - 3:
                pos.append((j, i)); break
        if len(pos) >= a.graded:
            break
    grid = [0.0, 0.5, 1.0, 1.5, 2.0]
    surf = {f"{sa},{sb}": {"d_ph": [], "d_true": []} for sa in grid for sb in grid}
    vec = {"joint_vs_vecsum_rel": [], "cos_single": []}
    for (j, i) in pos:
        p = prot[j]; ids = PAIR.ids(p["seq"]); nb = [i + o for o in (-3, -2, -1, 1, 2, 3)]
        xm = PAIR.masked(ids, nb)
        lp0 = PAIR.run(xm)
        D = {}
        for sa in grid:
            for sb in grid:
                lp = PAIR.run(xm, [(i, PAIR.a, "scale", sa), (i, PAIR.b, "scale", sb)])
                r = readouts(lp, lp0, nb, p["seq"])
                surf[f"{sa},{sb}"]["d_ph"].append(r["d_ph"]); surf[f"{sa},{sb}"]["d_true"].append(r["d_true"]); D[(sa, sb)] = r["vec"]
        da, db, dab = D[(0.0, 1.0)], D[(1.0, 0.0)], D[(0.0, 0.0)]
        vs = da + db
        vec["joint_vs_vecsum_rel"].append(float((dab - vs).norm() / max(float(dab.norm()), 1e-12)))
        vec["cos_single"].append(float(torch.nn.functional.cosine_similarity(da, db, dim=0)))
    disc["graded"] = {"positions": len(pos), "readout": "masked +-3 neighbours", "surface_median": {k: {m: round(st.median(v[m]), 5) for m in v} for k, v in surf.items()},
                      "joint_vs_vector_sum_rel_median": round(st.median(vec["joint_vs_vecsum_rel"]), 4), "cos_single_median": round(st.median(vec["cos_single"]), 4)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "discovery.json").write_text(json.dumps(disc, indent=1))
    print(json.dumps(disc, indent=1))


def cmd_confirm(a):
    global PAIR
    PAIR = Pair()
    F = json.load(open(a.frozen))
    alpha, beta = F["fit"]["alpha"], F["fit"]["beta"]; lo, hi = F["residual_q10_q90"]; qa, qb = F["q99"]
    first = json.load(open(a.exclude_json)) if a.exclude_json else []
    prot = proteins(a, True, a.n, a.seed, exclude=first)
    b3, _ = burial3_of(prot)
    H = activations(prot)
    ctx = F["predicted_context"]                       # {"departure": "b_heavy"|"a_heavy", "offset": o, "residues": "..."} or state/burial
    def in_ctx(p, b, i):
        if ctx["kind"] == "residue":
            j = i + ctx["offset"]; return 0 <= j < len(p["seq"]) and p["seq"][j] in ctx["residues"]
        if ctx["kind"] == "state5":
            return int(state5(p)[i]) in ctx["values"]
        if ctx["kind"] == "burial3":
            return int(b[i]) in ctx["values"]
        raise ValueError(ctx)
    # (a) recurrence
    tot = Counter()
    rows = []
    for j, (p, b, h) in enumerate(zip(prot, b3, H)):
        cand = (h[:, 0] >= qa) | (h[:, 1] >= qb)
        rr = h[:, 1] - (alpha + beta * h[:, 0])
        for i in torch.nonzero(cand).flatten().tolist():
            dep = "b_heavy" if rr[i] >= hi else ("a_heavy" if rr[i] <= lo else "mid")
            c = in_ctx(p, b, i)
            tot[(dep, c)] += 1
            if dep == ctx["departure"] and 3 <= i < len(p["seq"]) - 3:
                rows.append((j, i, c, float(h[i, 0]), float(h[i, 1])))
    n_dep = tot[(ctx["departure"], True)] + tot[(ctx["departure"], False)]
    n_all = sum(tot.values()); n_ctx = sum(v for (d, c), v in tot.items() if c)
    rec = {"departures": n_dep, "departures_in_context": tot[(ctx["departure"], True)], "share_in_context": tot[(ctx["departure"], True)] / max(n_dep, 1),
           "base_share_in_context": n_ctx / max(n_all, 1)}
    # (b) remove the natural variation at departure positions; perturbation-matched control in the same masked run
    g = torch.Generator().manual_seed(7)
    out = []
    for (j, i, c, ha, hb) in rows[: a.max_positions]:
        p = prot[j]; ids = PAIR.ids(p["seq"]); nb = [i + o for o in (-3, -2, -1, 1, 2, 3)]
        xm = PAIR.masked(ids, nb)
        lp0 = PAIR.run(xm)
        h_m = PAIR.cap["h"][i + 1].clone()                                      # pre-GELU in the masked forward
        g_b = float(torch.nn.functional.gelu(h_m[PAIR.b]))
        g_fit = float(torch.nn.functional.gelu(torch.tensor(alpha + beta * float(h_m[PAIR.a]))))
        delta = g_fit - g_b
        lp1 = PAIR.run(xm, [(i, PAIR.b, "set", g_fit)])
        r1 = readouts(lp1, lp0, nb, p["seq"])
        pert = abs(delta) * float(PAIR.wout[:, PAIR.b].norm())
        u = int(torch.randint(PAIR.wout.shape[1], (1,), generator=g))
        while u in (PAIR.a, PAIR.b):
            u = int(torch.randint(PAIR.wout.shape[1], (1,), generator=g))
        dc = pert / float(PAIR.wout[:, u].norm()) * (1 if torch.rand(1, generator=g) < 0.5 else -1)
        lpc = PAIR.run(xm, [(i, u, "add", dc)])
        rc = readouts(lpc, lp0, nb, p["seq"])
        out.append({"protein": p["acc"], "pos": i, "in_context": c, "delta_gelu_b": delta, "perturbation_norm": pert,
                    "d_true": r1["d_true"], "d_ph": r1["d_ph"], "ctl_d_true": rc["d_true"], "ctl_d_ph": rc["d_ph"]})
    def med(v):
        return round(st.median(v), 6) if v else None
    summ = {"recurrence": rec, "positions": len(out)}
    for c in (True, False):
        R = [r for r in out if r["in_context"] == c]
        summ[f"in_context={c}"] = {"n": len(R), "d_true_median": med([r["d_true"] for r in R]), "ctl_d_true_median": med([r["ctl_d_true"] for r in R]),
                                   "d_true_mean": round(st.mean([r["d_true"] for r in R]), 6) if R else None,
                                   "ctl_d_true_mean": round(st.mean([r["ctl_d_true"] for r in R]), 6) if R else None,
                                   "frac_hurt": round(sum(r["d_true"] < 0 for r in R) / max(len(R), 1), 3),
                                   "ctl_frac_hurt": round(sum(r["ctl_d_true"] < 0 for r in R) / max(len(R), 1), 3),
                                   "d_ph_median": med([r["d_ph"] for r in R]), "perturbation_norm_median": med([r["perturbation_norm"] for r in R])}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "confirm_cases.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
    (Path(a.out) / "confirm.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


POLAR_NB = "DEKRNQSTH"


def ols_boot(y, X, B=2000, seed=0):
    """OLS coefficients and bootstrap 95% intervals (rows resampled)."""
    y = torch.tensor(y, dtype=torch.float64); X = torch.tensor(X, dtype=torch.float64)
    fit = lambda yy, XX: torch.linalg.lstsq(XX, yy[:, None]).solution[:, 0]  # noqa: E731
    b0 = fit(y, X)
    g = torch.Generator().manual_seed(seed); n = len(y)
    bs = torch.stack([fit(y[i], X[i]) for i in (torch.randint(n, (n,), generator=g) for _ in range(B))])
    lo, hi = torch.quantile(bs, 0.025, dim=0), torch.quantile(bs, 0.975, dim=0)
    return [(round(float(b), 6), round(float(l), 6), round(float(h), 6)) for b, l, h in zip(b0, lo, hi)]


def cmd_confirm2(a):
    """The frozen test of results/pair_balance_frozen.md."""
    global PAIR
    PAIR = Pair()
    F = json.load(open(a.frozen))
    alpha, beta = F["fit"]["alpha"], F["fit"]["beta"]; lo, hi = F["residual_q10_q90"]; qa, qb = F["q99"]; qa9, qb9 = F["q99.9"]
    first = json.load(open(a.exclude_json)) if a.exclude_json else []
    prot = proteins(a, True, a.n, a.seed, exclude=first)
    b3, _ = burial3_of(prot)
    H = activations(prot)
    g = torch.Generator().manual_seed(11)
    nat, near, rec = [], [], Counter()
    for j, (p, h) in enumerate(zip(prot, H)):
        cand = (h[:, 0] >= qa) | (h[:, 1] >= qb)
        rr = h[:, 1] - (alpha + beta * h[:, 0])
        for i in torch.nonzero(cand).flatten().tolist():
            ds = p["seq"][i] in "DS"
            dep = rr[i] <= lo
            rec[(bool(dep), ds)] += 1
            if not (3 <= i < len(p["seq"]) - 3):
                continue
            if dep:
                nat.append((j, i, float(rr[i])))
            elif abs(float(rr[i])) <= 0.1:
                near.append((j, i, float(rr[i])))
    n_dep = rec[(True, True)] + rec[(True, False)]; n_all = sum(rec.values()); n_ds = rec[(True, True)] + rec[(False, True)]
    recurrence = {"departures": n_dep, "with_centre_DS": rec[(True, True)], "share": rec[(True, True)] / max(n_dep, 1),
                  "base_share": n_ds / max(n_all, 1), "ratio": (rec[(True, True)] / max(n_dep, 1)) / max(n_ds / max(n_all, 1), 1e-9)}
    borrow = [r for (_, _, r) in nat]

    def site(j, i, kind, r_borrow=None):
        p = prot[j]; ids = PAIR.ids(p["seq"]); nb = [i + o for o in (-3, -2, -1, 1, 2, 3)]
        xm = PAIR.masked(ids, nb)
        lp_nat = PAIR.run(xm)                                                   # the run as it is
        hm = PAIR.cap["h"][i + 1].clone()
        gb = float(torch.nn.functional.gelu(hm[PAIR.b]))
        if kind == "natural":
            alt = float(torch.nn.functional.gelu(torch.tensor(alpha + beta * float(hm[PAIR.a]))))   # departure removed
            lp_alt = PAIR.run(xm, [(i, PAIR.b, "set", alt)])
            r_with, r_without = readouts(lp_nat, lp_alt, nb, p["seq"]), None
            gain = r_with["d_true"]                                             # with departure minus without
            delta = gb - alt
        else:
            inj = float(torch.nn.functional.gelu(hm[PAIR.b] + r_borrow))        # departure injected
            lp_inj = PAIR.run(xm, [(i, PAIR.b, "set", inj)])
            gain = readouts(lp_inj, lp_nat, nb, p["seq"])["d_true"]             # with departure minus without
            delta = inj - gb
        pert = abs(delta) * float(PAIR.wout[:, PAIR.b].norm())
        u = int(torch.randint(PAIR.wout.shape[1], (1,), generator=g))
        while u in (PAIR.a, PAIR.b):
            u = int(torch.randint(PAIR.wout.shape[1], (1,), generator=g))
        dc = pert / float(PAIR.wout[:, u].norm()) * (1 if torch.rand(1, generator=g) < 0.5 else -1)
        ctl = readouts(PAIR.run(xm, [(i, u, "add", dc)]), lp_nat, nb, p["seq"])["d_true"]
        polar = sum(p["seq"][q] in POLAR_NB for q in nb) / 6
        return {"kind": kind, "protein": p["acc"], "pos": i, "centre": p["seq"][i], "centre_DS": int(p["seq"][i] in "DS"),
                "polar_frac": polar, "gain": gain, "ctl_abs_change": abs(ctl), "gain_abs": abs(gain), "perturbation_norm": pert}
    rows = []
    for (j, i, r) in nat[: a.max_positions]:
        rows.append(site(j, i, "natural"))
    pick = torch.randperm(len(near), generator=g)[: a.max_positions].tolist()
    for k in pick:
        j, i, _ = near[k]
        rows.append(site(j, i, "injected", borrow[int(torch.randint(len(borrow), (1,), generator=g))]))
    y = [r["gain"] for r in rows]
    X = [[1.0, float(r["kind"] == "natural"), r["polar_frac"], float(r["centre_DS"])] for r in rows]
    coef = ols_boot(y, X)
    # slopes on polarity within each kind
    sl = {}
    for kind in ("natural", "injected"):
        R = [r for r in rows if r["kind"] == kind]
        if len(R) > 5:
            sl[kind] = ols_boot([r["gain"] for r in R], [[1.0, r["polar_frac"]] for r in R])[1]
    # context transplant
    tsum = {}
    for ui, (u, q99u, q999u) in enumerate(((PAIR.a, qa, qa9), (PAIR.b, qb, qb9))):
        tops = [(j, i) for j, h in enumerate(H) for i in torch.nonzero(h[:, ui] >= q999u).flatten().tolist() if 3 <= i < len(prot[j]["seq"]) - 3]
        tops = [tops[k] for k in torch.randperm(len(tops), generator=g)[: a.transplants].tolist()]
        hosts = {s_: [] for s_ in ("strand", "helix", "loop")}
        statemap = {2: "strand", 1: "helix", 0: "loop"}
        for j, p in enumerate(prot):
            s5 = state5(p)
            for i in range(3, len(p["seq"]) - 3):
                if statemap.get(int(s5[i])):
                    hosts[statemap[int(s5[i])]].append((j, i))
        res_ = {s_: [] for s_ in hosts}
        for (j, i) in tops:
            win = prot[j]["seq"][i - 3:i + 4]
            for s_, hl in hosts.items():
                for k in torch.randint(len(hl), (5,), generator=g).tolist():
                    hj, hi_ = hl[k]
                    if hj == j:
                        continue
                    hs = prot[hj]["seq"]; new = hs[:hi_ - 3] + win + hs[hi_ + 4:]
                    PAIR.run(PAIR.ids(new))
                    res_[s_].append(float(PAIR.cap["h"][hi_ + 1, u]))
        tsum[str(u)] = {s_: {"n": len(v), "frac_above_q99": round(sum(x >= q99u for x in v) / max(len(v), 1), 3),
                             "median_h": round(st.median(v), 3) if v else None} for s_, v in res_.items()}
        tsum[str(u)]["sites"] = len(tops)
    summ = {"proteins": len(prot), "recurrence": recurrence,
            "sites": {"natural": sum(r["kind"] == "natural" for r in rows), "injected": sum(r["kind"] == "injected" for r in rows)},
            "regression_gain ~ 1 + natural + polar_frac + centre_DS (coef, 95% lo, hi)": dict(zip(["intercept", "natural", "polar_frac", "centre_DS"], coef)),
            "polar_slope_by_kind (coef, lo, hi)": sl,
            "gain_median": {k: round(st.median([r["gain"] for r in rows if r["kind"] == k]), 6) for k in ("natural", "injected") if any(r["kind"] == k for r in rows)},
            "abs_change_median_vs_matched_control": {k: [round(st.median([r["gain_abs"] for r in rows if r["kind"] == k]), 6),
                                                         round(st.median([r["ctl_abs_change"] for r in rows if r["kind"] == k]), 6)] for k in ("natural", "injected")
                                                     if any(r["kind"] == k for r in rows)},
            "transplant": tsum}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "confirm2_sites.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (Path(a.out) / "confirm2.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("discover", "confirm", "confirm2"):
        q = sp.add_parser(name)
        q.add_argument("--data", required=True); q.add_argument("--desc-dir", required=True); q.add_argument("--uniref", required=True)
        q.add_argument("--out", required=True); q.add_argument("--n", type=int, default=600); q.add_argument("--seed", type=int, default=2)
        if name == "discover":
            q.add_argument("--graded", type=int, default=30)
        else:
            q.add_argument("--frozen", required=True); q.add_argument("--exclude-json", default=None)
            q.add_argument("--max-positions", type=int, default=300)
            q.add_argument("--transplants", type=int, default=40)
    a = ap.parse_args()
    {"discover": cmd_discover, "confirm": cmd_confirm, "confirm2": cmd_confirm2}[a.cmd](a)


if __name__ == "__main__":
    main()
