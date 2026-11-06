"""650M layer 15 unit 2087: does changing this signal alter how the model uses repeat information?
Plan and frozen predictions: results/ubq650_use_plan.md.

  build : choose base windows (discovery: non-held-out, seed 2; fresh: held-out, seed 1) and write every construct,
          mask and target to a JSON file. No model is run. The file is the record of exactly what was tested.
  run   : execute the forwards of a constructs file (ESM-2 650M masked LM, fp32, TF32 off) and save raw responses:
            behaviour     log p over the 20 residues at the masked target; 2087 pre-GELU at every position; named
                          screen units at the target, its counterparts and the reference-block mean
            screen        unmasked 4-block constructs (c = 0..3 copies, the rest scrambles, constant length) and
                          masked c = 1, 2: named units at every reference position; layer-15 reference-block mean and
                          mean square for all 5,120 units
            intervention  repeat construct (recipient) and matched scramble construct (donor, same masks); layer-15
                          post-GELU replaced toward the donor (alpha 0.25 / 0.5 / 0.75 / 1) for 2087, then for a
                          frozen layer-15 group, against write-matched unit controls, random residual directions of
                          the same per-position norm, and random groups matched on combined residual change
  select-group : from a discovery run's screen, the layer-15 candidate group by the rule in the plan

  python scripts/ubq650_use.py build --set discovery --n-bases 8 --k 4 --out cons_disc.json
  python scripts/ubq650_use.py run --constructs cons_disc.json --parts behaviour,screen,intervention --group group.json --out dir
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

AA = "ACDEFGHIKLMNPQRSTVWY"
BL = 80                                  # block length
LAYER, UNIT = 15, 2087
DOSE_FRAGMENTS = {"P90745", "G2X1Q3", "Q4WN10", "Q9I485", "I1RLL3"}   # used in the earlier dose test
UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


# ----------------------------------------------------------------------------------------------- build
def ok_window(w):
    return set(w) <= set(AA) and len(set(w)) >= 12 and len({w[i:i + 6] for i in range(len(w) - 5)}) == len(w) - 5


def pick_bases(d, held, want_held, n, exclude, seed):
    rng = random.Random(seed)
    idx = [i for i, x in enumerate(d["acc"]) if (x in held) == want_held and x not in exclude and len(d["seq"][i]) >= 140]
    rng.shuffle(idx)
    out = []
    for i in idx:
        s = d["seq"][i]; st = rng.randint(20, len(s) - BL - 20); w = s[st:st + BL]
        if ok_window(w):
            out.append({"acc": d["acc"][i], "start": st, "R": w})
        if len(out) == n:
            break
    return out


def shuffled(w, rng):
    while True:
        x = list(w); rng.shuffle(x); x = "".join(x)
        if x != w:
            return x


def cmd_build(a):
    d = torch.load(a.data, weights_only=False)
    held = {x.split("\t")[0] for x in open(a.held) if not x.startswith("Entry")}
    ubq_accs = {x.split("\t")[0] for x in open(a.ubq_copies)}
    exclude = DOSE_FRAGMENTS | ubq_accs
    seed = 2 if a.set == "discovery" else 1
    bases = pick_bases(d, held, a.set == "fresh", a.n_bases, exclude, seed)
    vcount = {c: 0 for c in AA}; vrng = random.Random(seed + 100)
    fw = []                                                         # every forward of every part
    for b, base in enumerate(bases):
        R = base["R"]; rng = random.Random(seed * 1000 + b)
        base["targets"] = sorted(rng.sample(range(4, BL - 4), a.k))
        base["scr"] = [[shuffled(R, rng) for _ in range(3)] for _ in range(2)]   # draw d, block j
        base["v"] = []
        for p in base["targets"]:                                 # balanced visible residue for disagreement
            t = R[p]; cand = [c for c in AA if c != t]; m = min(vcount[c] for c in cand)
            v = vrng.choice([c for c in cand if vcount[c] == m]); vcount[v] += 1; base["v"].append(v)
        S = base["scr"]
        for it, (p, v) in enumerate(zip(base["targets"], base["v"])):
            D = R[:p] + v + R[p + 1:]
            for order in ("RX", "XR"):
                r0, x0 = (0, BL) if order == "RX" else (BL, 0)
                def two(X, mask_all, cond):
                    seq = R + X if order == "RX" else X + R
                    fw.append({"part": "behaviour", "base": b, "item": it, "cond": cond, "order": order, "seq": seq,
                               "ref": r0, "target": r0 + p, "counterparts": [x0 + p], "mask": [r0 + p] + ([x0 + p] if mask_all else [])})
                two(R, False, "copy_vis"); two(D, False, "dis_vis"); two(R, True, "copy_all")
                for dd in (0, 1):
                    two(S[dd][0], False, f"scr{dd}_vis"); two(S[dd][0], True, f"scr{dd}_all")
            fw.append({"part": "behaviour", "base": b, "item": it, "cond": "deletion", "order": "R", "seq": R, "ref": 0,
                       "target": p, "counterparts": [], "mask": [p]})
            fw.append({"part": "behaviour", "base": b, "item": it, "cond": "b4_c3", "order": "slot1", "seq": R * 4, "ref": 0,
                       "target": p, "counterparts": [BL + p, 2 * BL + p, 3 * BL + p], "mask": [p]})
            for dd in (0, 1):
                fw.append({"part": "behaviour", "base": b, "item": it, "cond": f"b4_c0_{dd}", "order": "slot1",
                           "seq": R + S[dd][0] + S[dd][1] + S[dd][2], "ref": 0, "target": p,
                           "counterparts": [BL + p, 2 * BL + p, 3 * BL + p], "mask": [p]})
        for slot in (1, 4):                                         # screen: constant length, c copies
            others = [2, 3, 4] if slot == 1 else [3, 2, 1]
            for c in (0, 1, 2, 3):
                for dd in ((0, 1) if c < 3 else (None,)):
                    blocks = {slot: R}; j = 0
                    for k_, sl in enumerate(others):
                        if k_ < c:
                            blocks[sl] = R
                        else:
                            blocks[sl] = S[dd][j]; j += 1
                    seq = "".join(blocks[s_] for s_ in (1, 2, 3, 4)); r0 = (slot - 1) * BL
                    fw.append({"part": "screen", "base": b, "item": None, "cond": f"c{c}" + ("" if dd is None else f"_{dd}"),
                               "order": f"slot{slot}", "seq": seq, "ref": r0, "target": None, "counterparts": [], "mask": []})
                    if slot == 1 and c in (1, 2):
                        for it, p in enumerate(base["targets"][:2]):
                            fw.append({"part": "screen", "base": b, "item": it, "cond": f"c{c}_{dd}_masked", "order": "slot1",
                                       "seq": seq, "ref": 0, "target": p, "counterparts": [BL + p, 2 * BL + p, 3 * BL + p], "mask": [p]})
    out = {"set": a.set, "seed": seed, "block": BL, "n_bases": len(bases), "k": a.k, "bases": bases,
           "v_counts": vcount, "t_counts": {c: sum(bs["R"][p] == c for bs in bases for p in bs["targets"]) for c in AA},
           "forwards": fw}
    Path(a.out).write_text(json.dumps(out))
    print(json.dumps({"set": a.set, "bases": len(bases), "accessions": [bs["acc"] for bs in bases][:60],
                      "forwards": {p: sum(f["part"] == p for f in fw) for p in ("behaviour", "screen")}}))


# ----------------------------------------------------------------------------------------------- model
def named_units(units_json, seed=20261106):
    ex = json.load(open(units_json))
    cols = [(e["layer"], e["unit"], e["pole"], "ubq") for e in ex]
    taken = {(l, u) for l, u, _, _ in cols}; rng = random.Random(seed)
    for l in range(33):
        pool = [u for u in range(5120) if (l, u) not in taken]
        cols += [(l, u, 0, "comparison") for u in sorted(rng.sample(pool, 4))]
    return cols


class M650:
    def __init__(self, device, cols):
        from transformers import EsmForMaskedLM, EsmTokenizer
        hf = "facebook/esm2_t33_650M_UR50D"
        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        self.tok = EsmTokenizer.from_pretrained(hf)
        self.m = EsmForMaskedLM.from_pretrained(hf).float().to(device).eval(); self.dev = device
        self.aa = torch.tensor([self.tok.convert_tokens_to_ids(c) for c in AA], device=device)
        self.cols = cols
        self.by_layer = {l: torch.tensor([u for (ll, u, _, _) in cols if ll == l], device=device) for l in range(33)}
        self.col_pos = {l: [i for i, (ll, _, _, _) in enumerate(cols) if ll == l] for l in range(33)}
        lay = self.m.esm.encoder.layer
        self.wout = lay[LAYER].output.dense.weight.detach()          # [1280, 5120]
        self.st = {}
        for l in range(33):
            lay[l].intermediate.dense.register_forward_hook(self._pre(l))
        lay[LAYER].intermediate.register_forward_hook(self._post)
        lay[LAYER].output.dense.register_forward_hook(self._add)

    def _pre(self, l):
        def h(_m, _i, o):
            self.st[("pre", l)] = o[0, :, self.by_layer[l]]
            if l == LAYER:
                self.st["pre15"] = o[0]
        return h

    def _post(self, _m, _i, o):
        self.st["post15"] = o[0].detach().clone()
        p = self.st.get("patch")
        if p is not None:
            pos, units, alpha, donor = p
            o = o.clone(); cur = o[0][pos][:, units]
            o[0, pos[:, None], units[None, :]] = (1 - alpha) * cur + alpha * donor[pos][:, units]
        return o

    def _add(self, _m, _i, o):
        a = self.st.get("add")
        if a is not None:
            pos, vec = a
            o = o.clone(); o[0, pos] += vec
        return o

    def fwd(self, f, patch=None, add=None):
        ids = self.tok(f["seq"], return_tensors="pt")["input_ids"].to(self.dev)
        for q in f["mask"]:
            ids[0, q + 1] = self.tok.mask_token_id
        self.st = {"patch": patch, "add": add}
        with torch.no_grad():
            logits = self.m(input_ids=ids).logits[0]
        out = {}
        if f["target"] is not None:
            out["lp"] = torch.log_softmax(logits[f["target"] + 1].double(), -1)[self.aa].float().cpu()
        named = torch.empty(len(self.cols), ids.shape[1], device=self.dev)
        for l in range(33):
            if self.col_pos[l]:
                named[self.col_pos[l]] = self.st[("pre", l)].T
        out["named_all"] = named[:, 1:-1]                              # [n_cols, L] (device)
        out["u2087"] = self.st["pre15"][1:-1, UNIT].float().cpu().clone()   # clone: on CPU a view would pickle all of pre15
        return out


def summarise_named(o, f):
    """named units at the target, up to 3 counterparts, and the reference-block mean (NaN-padded)."""
    na = o["named_all"]; n = na.shape[0]; r = f["ref"]
    cols = [na[:, f["target"]] if f["target"] is not None else torch.full((n,), float("nan"), device=na.device)]
    cps = f["counterparts"] + [None] * (3 - len(f["counterparts"]))
    cols += [na[:, c] if c is not None else torch.full((n,), float("nan"), device=na.device) for c in cps]
    cols += [na[:, r:r + BL].mean(1)]
    return torch.stack(cols, 1).half().cpu()                         # [n, 5]


# ----------------------------------------------------------------------------------------------- run parts
def run_behaviour(M, fws, log):
    recs = []
    for i, f in enumerate(fws):
        o = M.fwd(f)
        recs.append({"lp": o["lp"], "u2087": o["u2087"], "named": summarise_named(o, f)})
        if i % 200 == 0:
            log(f"behaviour {i}/{len(fws)}")
    return recs


def run_screen(M, fws, log):
    recs = []
    for i, f in enumerate(fws):
        o = M.fwd(f); r = f["ref"]
        pre = M.st["pre15"][1:-1][r:r + BL].float()                   # [80, 5120]
        rec = {"named_ref": o["named_all"][:, r:r + BL].half().cpu(), "u2087": o["u2087"],
               "l15_mean": pre.mean(0).cpu(), "l15_msq": (pre ** 2).mean(0).cpu()}
        if f["target"] is not None:
            rec["lp"] = o["lp"]
        recs.append(rec)
        if i % 100 == 0:
            log(f"screen {i}/{len(fws)}")
    return recs


def select_group(cons, screen_recs, n=20):
    """Rule (plan): z_u = mean over bases of [ref-block mean at c = 3 - mean over draws at c = 0], both slots averaged,
    divided by the pooled within-forward-and-across-forward SD of the unit over reference positions in all c = 0
    forwards. Group = 2087 plus the 19 other layer-15 units with the largest positive z."""
    fws = [f for f in cons["forwards"] if f["part"] == "screen"]
    nb = cons["n_bases"]; diff = torch.zeros(nb, 5120); cnt = torch.zeros(nb)
    m0, s0, k0 = torch.zeros(5120, dtype=torch.float64), torch.zeros(5120, dtype=torch.float64), 0
    by = {}
    for f, r in zip(fws, screen_recs):
        if f["target"] is None:
            by[(f["base"], f["order"], f["cond"])] = r
    for b in range(nb):
        for slot in ("slot1", "slot4"):
            c3 = by[(b, slot, "c3")]["l15_mean"]; c0 = (by[(b, slot, "c0_0")]["l15_mean"] + by[(b, slot, "c0_1")]["l15_mean"]) / 2
            diff[b] += c3 - c0; cnt[b] += 1
            for dd in (0, 1):
                r = by[(b, slot, f"c0_{dd}")]; m0 += r["l15_mean"].double(); s0 += r["l15_msq"].double(); k0 += 1
    resp = (diff / cnt[:, None]).mean(0)
    sd = (s0 / k0 - (m0 / k0) ** 2).clamp(min=1e-12).sqrt().float()
    z = resp / sd
    order = [u for u in torch.argsort(z, descending=True).tolist() if u != UNIT and z[u] > 0][: n - 1]
    return {"rule": select_group.__doc__.strip(), "group": [UNIT] + order, "z": {str(u): round(float(z[u]), 3) for u in [UNIT] + order},
            "z_2087": round(float(z[UNIT]), 3), "rank_2087": int((z > z[UNIT]).sum()) + 1,
            "z_quantiles_all_units": [round(float(torch.quantile(z, q)), 3) for q in (0.5, 0.9, 0.99, 0.999)]}


def run_intervention(M, cons, group, log, seed=7):
    fwb = [f for f in cons["forwards"] if f["part"] == "behaviour"]
    key = {(f["base"], f["item"], f["cond"], f["order"]): f for f in fwb}
    G = torch.tensor(group, device=M.dev); wn = M.wout.norm(dim=0)          # [5120]
    excl = set(group) | {UNIT}
    pool = torch.tensor([u for u in range(5120) if u not in excl], device=M.dev)
    recs = []; n_items = sum(len(b["targets"]) for b in cons["bases"]); done = 0
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for b, base in enumerate(cons["bases"]):
        for it in range(len(base["targets"])):
            for rname, rc, dc, order in (("agree", "copy_vis", "scr0_vis", "RX"), ("disagree", "dis_vis", "scr0_vis", "RX"),
                                        ("b4_c3", "b4_c3", "b4_c0_0", "slot1")):
                fr, fd = key[(b, it, rc, order)], key[(b, it, dc, order)]
                od = M.fwd(fd); donor = M.st["post15"]
                orr = M.fwd(fr); rec_post = M.st["post15"]
                T = rec_post.shape[0]; L_ = T - 2
                allpos = torch.arange(1, T - 1, device=M.dev)
                refpos = torch.arange(1 + fr["ref"], 1 + fr["ref"] + BL, device=M.dev)
                other = torch.tensor([q for q in range(1, T - 1) if not (1 + fr["ref"] <= q < 1 + fr["ref"] + BL)], device=M.dev)
                dg = (donor - rec_post)[allpos]                         # [L, 5120]
                n_u = dg.norm(dim=0) * wn                               # change norm per unit over all positions
                cand = [u for u in torch.argsort((n_u - n_u[UNIT]).abs()).tolist() if u not in excl][:4]
                per_pos = dg[:, UNIT].abs() * wn[UNIT]                  # per-position norm of 2087's change [L]
                Wg = M.wout[:, G]; gvec = dg[:, G] @ Wg.T; NG = gvec.norm()       # combined change, cancellation included
                lvec = dg @ M.wout.T                                                 # whole layer-15 FFN change [L, 1280]
                # matched group controls (amended after the pilot, where random groups reached only 0.37-0.57 of the
                # group's combined change): draw random same-size groups from all non-group units, then from the
                # top-k non-group units by single-unit change norm (k = 2000, 1000, 500, 200), until 4 groups fall
                # within +-10% of the group's combined norm; otherwise the 4 closest over all draws
                order_nu = [u for u in torch.argsort(n_u, descending=True).tolist() if u not in excl]
                pools = [("all", pool)] + [(f"top{k}", torch.tensor(order_nu[:k], device=M.dev)) for k in (2000, 1000, 500, 200)]
                found, tried = [], []
                for pname, pl in pools:
                    sel = torch.stack([pl[torch.randperm(len(pl), generator=gen)[:len(group)].to(M.dev)] for _ in range(200)])
                    rat = torch.stack([(dg[:, s] @ M.wout[:, s].T).norm() for s in sel]).cpu() / float(NG)
                    tried += [(float(rat[i]), sel[i], pname) for i in range(200)]
                    found += [(float(rat[i]), sel[i], pname) for i in range(200) if abs(float(rat[i]) - 1) <= 0.10]
                    if len(found) >= 4:
                        break
                if len(found) < 4:
                    found = sorted(tried, key=lambda x: abs(x[0] - 1))[:4]
                found = found[:4]
                ctrl_groups = [x[1] for x in found]; ratios = [round(x[0], 3) for x in found]; pools_used = [x[2] for x in found]
                U = torch.tensor([UNIT], device=M.dev)
                variants = [("u2087_a0.25", (allpos, U, 0.25, donor), None), ("u2087_a0.5", (allpos, U, 0.5, donor), None),
                            ("u2087_a0.75", (allpos, U, 0.75, donor), None), ("u2087_a1", (allpos, U, 1.0, donor), None),
                            ("u2087_a1_refblock", (refpos, U, 1.0, donor), None), ("u2087_a1_otherblocks", (other, U, 1.0, donor), None)]
                variants += [(f"unitctl{k}_{u}", (allpos, torch.tensor([u], device=M.dev), 1.0, donor), None) for k, u in enumerate(cand)]
                for k in range(2):
                    v = torch.randn(L_, M.wout.shape[0], generator=gen).to(M.dev)
                    v = v / v.norm(dim=1, keepdim=True) * per_pos[:, None]
                    variants.append((f"randdir{k}", None, (allpos, v)))
                variants += [("group_a0.5", (allpos, G, 0.5, donor), None), ("group_a1", (allpos, G, 1.0, donor), None)]
                variants += [(f"groupctl{k}", (allpos, s, 1.0, donor), None) for k, s in enumerate(ctrl_groups)]
                for k in range(2):                                              # group-level random directions
                    v = torch.randn(L_, M.wout.shape[0], generator=gen).to(M.dev)
                    variants.append((f"grouprand{k}", None, (allpos, v / v.norm(dim=1, keepdim=True) * gvec.norm(dim=1, keepdim=True))))
                ALL = torch.arange(5120, device=M.dev)                          # secondary: whole layer-15 FFN output
                variants += [("layer15_a0.5", (allpos, ALL, 0.5, donor), None), ("layer15_a1", (allpos, ALL, 1.0, donor), None)]
                for k in range(2):
                    v = torch.randn(L_, M.wout.shape[0], generator=gen).to(M.dev)
                    variants.append((f"layerrand{k}", None, (allpos, v / v.norm(dim=1, keepdim=True) * lvec.norm(dim=1, keepdim=True))))
                lps = []
                for name, patch, add in variants:
                    lps.append(M.fwd(fr, patch=patch, add=add)["lp"])
                recs.append({"base": b, "item": it, "recipient": rname, "rec_cond": rc, "donor_cond": dc, "order": order,
                             "lp_rec": orr["lp"], "lp_don": od["lp"], "variants": [v[0] for v in variants], "lp_var": torch.stack(lps),
                             "n_2087": float(n_u[UNIT]), "unit_controls": cand, "n_unit_controls": [float(n_u[u]) for u in cand],
                             "group_norm": float(NG), "group_control_ratios": ratios, "group_control_pools": pools_used,
                             "layer15_norm": float(lvec.norm()),
                             "u2087_rec": orr["u2087"], "u2087_don": od["u2087"], "patched_positions": "all residue tokens"})
            done += 1
            if done % 10 == 0:
                log(f"intervention {done}/{n_items} items")
    return recs


def cmd_run(a):
    t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "progress.log", "a")

    def log(m):
        s = f"[{time.strftime('%H:%M:%S')}] {m}"; print(s, flush=True); logf.write(s + "\n"); logf.flush()
    torch.set_grad_enabled(False)
    if a.device == "cpu":
        torch.set_num_threads(a.threads)
    cons = json.load(open(a.constructs)); cols = named_units(a.units)
    M = M650(a.device, cols)
    meta = {"constructs": a.constructs, "device": a.device, "torch": torch.__version__, "named_columns": cols,
            "cuda": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None}
    parts = a.parts.split(",")
    if "behaviour" in parts:
        fws = [f for f in cons["forwards"] if f["part"] == "behaviour"]
        t = time.time(); recs = run_behaviour(M, fws, log)
        torch.save({"meta": meta, "forwards": fws, "recs": recs}, out / "behaviour.pt"); log(f"behaviour done {time.time() - t:.0f}s")
    if "screen" in parts:
        fws = [f for f in cons["forwards"] if f["part"] == "screen"]
        t = time.time(); recs = run_screen(M, fws, log)
        torch.save({"meta": meta, "forwards": fws, "recs": recs}, out / "screen.pt"); log(f"screen done {time.time() - t:.0f}s")
        if a.group is None:
            g = select_group(cons, [r for r in recs])
            (out / "group_selected.json").write_text(json.dumps(g, indent=1)); log(f"group selected: {g['group']}")
    if "intervention" in parts:
        group = json.load(open(a.group if a.group else out / "group_selected.json"))["group"]
        t = time.time(); recs = run_intervention(M, cons, group, log)
        torch.save({"meta": meta, "group": group, "recs": recs}, out / "intervention.pt"); log(f"intervention done {time.time() - t:.0f}s")
    (out / "run_done.json").write_text(json.dumps({"seconds": round(time.time() - t0, 1), "parts": parts}))
    log(f"all done {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--set", choices=["discovery", "fresh"], required=True); b.add_argument("--n-bases", type=int, required=True)
    b.add_argument("--k", type=int, required=True); b.add_argument("--out", required=True)
    b.add_argument("--data", default="data/swissprot_2026_03.pt"); b.add_argument("--held", default="lists/heldout_created_after_2021-11-17.tsv")
    b.add_argument("--ubq-copies", default="results/batch_20261105/ubiquitin_copies.tsv")
    r = sp.add_parser("run")
    r.add_argument("--constructs", required=True); r.add_argument("--parts", default="behaviour,screen,intervention")
    r.add_argument("--units", default="infra/runs/atlas-ru-20261105-1704/ubq650_units.json"); r.add_argument("--group", default=None)
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); r.add_argument("--threads", type=int, default=4)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
