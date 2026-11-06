"""Analysis for scripts/ubq650_use.py runs (plan: results/ubq650_use_plan.md). Uncertainty is across base sequences:
every quantity is first averaged within a base (over its targets, orders and scramble draws), then bootstrapped over
bases (2,000 draws); the number of bases with a positive mean is reported alongside. Per-item values are written
out in full (items.csv, intervention_items.csv) so nothing is only available as a summary.

  python scripts/ubq650_use_analyse.py --constructs cons.json --run dir --out dir
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

AA = "ACDEFGHIKLMNPQRSTVWY"
BL = 80
Q99, Q999 = 0.57, 2.19                       # 2087's atlas q99 / q99.9 edges (pre-GELU)


def boot(per_base, B=2000, seed=0):
    v = torch.tensor([x for x in per_base if x is not None and not math.isnan(x)], dtype=torch.float64)
    if len(v) == 0:
        return None
    g = torch.Generator().manual_seed(seed)
    bs = v[torch.randint(len(v), (B, len(v)), generator=g)].mean(1)
    return {"mean": round(float(v.mean()), 4), "median": round(float(v.median()), 4), "ci95": [round(float(torch.quantile(bs, 0.025)), 4),
            round(float(torch.quantile(bs, 0.975)), 4)], "bases": len(v), "bases_positive": int((v > 0).sum())}


def boot_ratio(num_by_base, den_by_base, B=2000, seed=0):
    """ratio of sums over bases (stable when per-item denominators are small), bootstrapped over bases."""
    n = torch.tensor(num_by_base, dtype=torch.float64); d = torch.tensor(den_by_base, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed); idx = torch.randint(len(n), (B, len(n)), generator=g)
    bs = n[idx].sum(1) / d[idx].sum(1)
    return {"ratio": round(float(n.sum() / d.sum()), 4), "ci95": [round(float(torch.quantile(bs, 0.025)), 4), round(float(torch.quantile(bs, 0.975)), 4)]}


def ent(lp):
    p = lp.double().exp(); p = p / p.sum()
    return float(-(p * p.clamp(min=1e-30).log()).sum())


def kl(lp_a, lp_b):
    a = lp_a.double().exp(); a = a / a.sum(); b = lp_b.double().exp(); b = b / b.sum()
    return float((a * (a.clamp(min=1e-30).log() - b.clamp(min=1e-30).log())).sum())


def behaviour(cons, run, out):
    D = torch.load(Path(run) / "behaviour.pt", weights_only=False)
    rec = {(f["base"], f["item"], f["cond"], f["order"]): r for f, r in zip(D["forwards"], D["recs"])}
    fkey = {(f["base"], f["item"], f["cond"], f["order"]): f for f in D["forwards"]}
    rows = []
    for b, base in enumerate(cons["bases"]):
        for it, (p, v) in enumerate(zip(base["targets"], base["v"])):
            t = base["R"][p]; ti, vi = AA.index(t), AA.index(v)
            lt = lambda c, o: float(rec[(b, it, c, o)]["lp"][ti])  # noqa: E731
            lv = lambda c, o: float(rec[(b, it, c, o)]["lp"][vi])  # noqa: E731
            row = {"base": b, "acc": base["acc"], "item": it, "pos": p, "t": t, "v": v}
            for o in ("RX", "XR"):
                scr_t = (lt("scr0_vis", o) + lt("scr1_vis", o)) / 2
                scr_pref = ((lv("scr0_vis", o) - lt("scr0_vis", o)) + (lv("scr1_vis", o) - lt("scr1_vis", o))) / 2
                row[f"agree_{o}"] = lt("copy_vis", o) - scr_t
                row[f"true_under_dis_{o}"] = lt("dis_vis", o) - scr_t
                row[f"pref_shift_{o}"] = (lv("dis_vis", o) - lt("dis_vis", o)) - scr_pref
                row[f"masked_all_{o}"] = lt("copy_all", o) - (lt("scr0_all", o) + lt("scr1_all", o)) / 2
                row[f"p_t_scr_{o}"] = math.exp(scr_t); row[f"p_t_copy_{o}"] = math.exp(lt("copy_vis", o))
                row[f"pmax_scr_{o}"] = float(rec[(b, it, "scr0_vis", o)]["lp"].exp().max())
                row[f"entropy_scr_{o}"] = ent(rec[(b, it, "scr0_vis", o)]["lp"]); row[f"entropy_copy_{o}"] = ent(rec[(b, it, "copy_vis", o)]["lp"])
                for c in ("copy_vis", "scr0_vis", "copy_all", "dis_vis"):
                    f_ = fkey[(b, it, c, o)]
                    u = rec[(b, it, c, o)]["u2087"]
                    row[f"u2087_target_{c}_{o}"] = float(u[f_["target"]])
                    row[f"u2087_counterpart_{c}_{o}"] = float(u[f_["counterparts"][0]])
                    row[f"u2087_refmean_{c}_{o}"] = float(u[f_["ref"]:f_["ref"] + BL].mean())
            row["deletion_vs_copy"] = (lt("copy_vis", "RX") + lt("copy_vis", "XR")) / 2 - lt("deletion", "R")
            row["deletion_vs_scr"] = (lt("scr0_vis", "RX") + lt("scr1_vis", "RX") + lt("scr0_vis", "XR") + lt("scr1_vis", "XR")) / 4 - lt("deletion", "R")
            row["b4_c3_vs_c0"] = lt("b4_c3", "slot1") - (lt("b4_c0_0", "slot1") + lt("b4_c0_1", "slot1")) / 2
            row["u2087_target_b4_c3"] = float(rec[(b, it, "b4_c3", "slot1")]["u2087"][p])
            row["u2087_target_b4_c0"] = (float(rec[(b, it, "b4_c0_0", "slot1")]["u2087"][p]) + float(rec[(b, it, "b4_c0_1", "slot1")]["u2087"][p])) / 2
            for k in ("agree", "true_under_dis", "pref_shift", "masked_all", "p_t_scr", "p_t_copy", "pmax_scr", "entropy_scr", "entropy_copy"):
                row[k] = (row[f"{k}_RX"] + row[f"{k}_XR"]) / 2
            rows.append(row)
    with open(Path(out) / "items.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    keys = ["agree", "true_under_dis", "pref_shift", "masked_all", "deletion_vs_copy", "deletion_vs_scr", "b4_c3_vs_c0",
            "agree_RX", "agree_XR", "pref_shift_RX", "pref_shift_XR", "p_t_scr", "p_t_copy", "pmax_scr", "entropy_scr", "entropy_copy",
            "u2087_target_copy_vis_RX", "u2087_target_scr0_vis_RX", "u2087_counterpart_copy_vis_RX", "u2087_counterpart_scr0_vis_RX",
            "u2087_refmean_copy_vis_RX", "u2087_refmean_scr0_vis_RX", "u2087_target_copy_all_RX", "u2087_target_dis_vis_RX",
            "u2087_target_b4_c3", "u2087_target_b4_c0"]
    nb = len(cons["bases"]); res = {}
    for k in keys:
        per = [sum(r[k] for r in rows if r["base"] == b) / max(1, sum(1 for r in rows if r["base"] == b)) for b in range(nb)]
        res[k] = boot(per)
    qs = lambda k: [round(float(torch.quantile(torch.tensor([r[k] for r in rows], dtype=torch.float64), q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)]  # noqa: E731
    res["item_quantiles (q10 q25 q50 q75 q90)"] = {k: qs(k) for k in ("agree", "pref_shift", "true_under_dis", "masked_all", "b4_c3_vs_c0", "p_t_scr", "p_t_copy")}
    res["items"] = len(rows)
    return res, rows


def screen(cons, run):
    D = torch.load(Path(run) / "screen.pt", weights_only=False)
    cols = D["meta"]["named_columns"]; fws = D["forwards"]; recs = D["recs"]
    by = {(f["base"], f["order"], f["cond"]): r for f, r in zip(fws, recs) if f["target"] is None}
    nb = cons["n_bases"]; n = len(cols)
    lvl = {c: torch.zeros(nb, n) for c in range(4)}
    m0 = torch.zeros(n, dtype=torch.float64); s0 = torch.zeros(n, dtype=torch.float64); k0 = 0
    for b in range(nb):
        for slot in ("slot1", "slot4"):
            for c in range(4):
                rs = [by[(b, slot, "c3")]] if c == 3 else [by[(b, slot, f"c{c}_0")], by[(b, slot, f"c{c}_1")]]
                lvl[c][b] += sum(r["named_ref"].float().mean(1) for r in rs) / len(rs) / 2
                if c == 0:
                    for r in rs:
                        x = r["named_ref"].double(); m0 += x.mean(1); s0 += (x ** 2).mean(1); k0 += 1
    sd = (s0 / k0 - (m0 / k0) ** 2).clamp(min=1e-12).sqrt().float()
    diff = lvl[3] - lvl[0]                                             # [nb, n]
    z_base = diff / sd; z = z_base.mean(0)
    sign = torch.tensor([-1.0 if p == 1 else 1.0 for (_, _, p, _) in cols])
    za = z * sign
    table = []
    for i, (l, u, p, kind) in enumerate(cols):
        table.append({"layer": l, "unit": u, "pole": p, "kind": kind, "z_c3_vs_c0": round(float(z[i]), 3),
                      "z_pole_adjusted": round(float(za[i]), 3), "bases_same_sign": int(((z_base[:, i] * sign[i]) > 0).sum()),
                      "dose_z (c0..c3)": [round(float(((lvl[c][:, i] - lvl[0][:, i]).mean()) / sd[i]), 3) for c in range(4)]})
    kinds = {}
    for kind in ("ubq", "comparison"):
        v = torch.tensor([t["z_pole_adjusted"] for t in table if t["kind"] == kind])
        kinds[kind] = {"n": len(v), "median_z": round(float(v.median()), 3), "q90_z": round(float(torch.quantile(v, 0.9)), 3),
                       "n_z_gt_2": int((v > 2).sum()), "n_z_gt_5": int((v > 5).sum())}
    # 2087 per-position profile on the reference block, unmasked, slot 1
    prof = {}
    for c in ("c0_0", "c0_1", "c1_0", "c2_0", "c3"):
        vals = torch.stack([by[(b, "slot1", c)]["u2087"][0:BL] for b in range(nb)])          # [nb, 80]
        prof[c] = {"ref_mean": round(float(vals.mean()), 3), "share_ref_positions_gt_q99": round(float((vals > Q99).double().mean()), 3),
                   "share_gt_q99.9": round(float((vals > Q999).double().mean()), 3), "ref_max_mean_over_bases": round(float(vals.max(1).values.mean()), 3)}
    resid = defaultdict(int); tot = defaultdict(int)
    for b in range(nb):
        vals = by[(b, "slot1", "c3")]["u2087"][0:BL]; R = cons["bases"][b]["R"]
        for i in range(BL):
            tot[R[i]] += 1; resid[R[i]] += int(vals[i] > Q999)
    prof["c3: share of reference positions above q99.9, by residue (n)"] = {k: [round(resid[k] / tot[k], 3), tot[k]] for k in sorted(tot, key=lambda x: -resid[x] / tot[x])[:8]}
    # masked dose for 2087 at reference positions (first two targets): c0 / c3 from behaviour, c1 / c2 from screen
    md = defaultdict(list)
    for f, r in zip(fws, recs):
        if f["target"] is not None:
            md[f["cond"].split("_")[0]].append(float(r["u2087"][0:BL].mean()))
    masked = {k: round(sum(v) / len(v), 3) for k, v in sorted(md.items())}
    return {"columns": table, "by_kind": kinds, "u2087_profile_unmasked": prof, "u2087_refmean_masked_c1_c2": masked}


def intervention(cons, run, out):
    D = torch.load(Path(run) / "intervention.pt", weights_only=False)
    rows = []
    for r in D["recs"]:
        base = cons["bases"][r["base"]]; p = base["targets"][r["item"]]; t = base["R"][p]; v = base["v"][r["item"]]
        ti, vi = AA.index(t), AA.index(v)
        lr, ld, lv = r["lp_rec"], r["lp_don"], r["lp_var"]
        row = {"base": r["base"], "item": r["item"], "recipient": r["recipient"], "t": t, "v": v,
               "lp_t_rec": float(lr[ti]), "lp_t_don": float(ld[ti]), "pref_rec": float(lr[vi] - lr[ti]), "pref_don": float(ld[vi] - ld[ti]),
               "n_2087": r["n_2087"], "group_ctl_ratio_mean": sum(r["group_control_ratios"]) / len(r["group_control_ratios"]),
               "u2087_target_rec": float(r["u2087_rec"][p]), "u2087_target_don": float(r["u2087_don"][p]),
               "u2087_mean_rec": float(r["u2087_rec"].mean()), "u2087_mean_don": float(r["u2087_don"].mean())}
        for k, name in enumerate(r["variants"]):
            nm = name.split("_")[0] if name.startswith("unitctl") else name
            row[f"d_lp_t:{nm}"] = float(lv[k][ti] - lr[ti]); row[f"d_pref:{nm}"] = float((lv[k][vi] - lv[k][ti]) - (lr[vi] - lr[ti]))
            row[f"kl:{nm}"] = kl(lr, lv[k])
        for pref_, nm in (("unitctl", "unitctl_mean"), ("randdir", "randdir_mean"), ("groupctl", "groupctl_mean"),
                          ("grouprand", "grouprand_mean"), ("layerrand", "layerrand_mean")):
            ks = [c.split(":", 1)[1] for c in row if c.startswith("d_lp_t:" + pref_) and not c.endswith("_mean")]
            if not ks:
                continue
            for q in ("d_lp_t", "d_pref", "kl"):
                row[f"{q}:{nm}"] = sum(row[f"{q}:{k}"] for k in ks) / len(ks)
        rows.append(row)
    with open(Path(out) / "intervention_items.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    nb = cons["n_bases"]; res = {"group": D["group"]}
    variants = [c.split(":", 1)[1] for c in rows[0] if c.startswith("d_lp_t:")]
    for rc in ("agree", "b4_c3", "disagree"):
        R_ = [x for x in rows if x["recipient"] == rc]; part = {}
        for nm in variants:
            if rc == "disagree":
                num = [sum(-x[f"d_pref:{nm}"] for x in R_ if x["base"] == b) for b in range(nb)]
                den = [sum(x["pref_rec"] - x["pref_don"] for x in R_ if x["base"] == b) for b in range(nb)]
            else:
                num = [sum(x[f"d_lp_t:{nm}"] for x in R_ if x["base"] == b) for b in range(nb)]
                den = [sum(x["lp_t_don"] - x["lp_t_rec"] for x in R_ if x["base"] == b) for b in range(nb)]
            per_dlp = [sum(x[f"d_lp_t:{nm}"] for x in R_ if x["base"] == b) / max(1, sum(1 for x in R_ if x["base"] == b)) for b in range(nb)]
            per_kl = [sum(x[f"kl:{nm}"] for x in R_ if x["base"] == b) / max(1, sum(1 for x in R_ if x["base"] == b)) for b in range(nb)]
            part[nm] = {"fraction_removed (ratio of sums)": boot_ratio(num, den), "d_lp_true (nats, per-base mean)": boot(per_dlp),
                        "kl_rec_to_patched": boot(per_kl)}
        part["_recipient"] = {"lp_t_rec": boot([sum(x["lp_t_rec"] for x in R_ if x["base"] == b) / max(1, sum(1 for x in R_ if x["base"] == b)) for b in range(nb)]),
                              "advantage (lp_t_rec - lp_t_don)": boot([sum(x["lp_t_rec"] - x["lp_t_don"] for x in R_ if x["base"] == b) / max(1, sum(1 for x in R_ if x["base"] == b)) for b in range(nb)]),
                              "u2087_target_rec / don": [round(sum(x["u2087_target_rec"] for x in R_) / len(R_), 3), round(sum(x["u2087_target_don"] for x in R_) / len(R_), 3)],
                              "u2087_mean_rec / don": [round(sum(x["u2087_mean_rec"] for x in R_) / len(R_), 3), round(sum(x["u2087_mean_don"] for x in R_) / len(R_), 3)],
                              "group_control_norm_ratio_mean": round(sum(x["group_ctl_ratio_mean"] for x in R_) / len(R_), 3)}
        res[rc] = part
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constructs", required=True); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cons = json.load(open(a.constructs)); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {}
    if (Path(a.run) / "behaviour.pt").exists():
        res["behaviour"], _ = behaviour(cons, a.run, out)
    if (Path(a.run) / "screen.pt").exists():
        res["screen"] = screen(cons, a.run)
    if (Path(a.run) / "intervention.pt").exists():
        res["intervention"] = intervention(cons, a.run, out)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "screen"}, indent=1)[:20000])


if __name__ == "__main__":
    main()
