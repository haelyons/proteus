"""Unit-specific tests for the round-2 8M cases (predictions: results/cases_round2_predictions.md).

Every test runs on --set discovery (non-held-out, seed 2; inspection and smoke) or --set fresh (held-out with
descriptors excluding the first experiment's 400, seed 1; or, for `shsp`, every held-out protein). Thresholds are
always the unit's discovery q99.9 (passed in, never re-estimated on fresh proteins).

  tm     5,1181  hydrophobic segment vs buried helix vs membrane: rate / median by category within KD19 bins
  shsp   3,1238  small-heat-shock-protein family: fresh family members (MMseqs2 to canonical seeds), CS-domain
                 fold analogues, broadcast (shuffle one part, read the other), +-10 window transplants, shuffles
  gc     5,668   whole-protein composition: per-protein mean vs GARP - FYMINK, variance shares, shuffles, chimeras
  nterm  1,746   N-terminal hydrophobic segment: SIGNAL / N-terminal TM / internal TM within KD19 bins; moving
                 signal peptides inward and internal TM segments to the N-terminus; next-layer readers
  tok    0,767 / 0,789  layer-0 units: centre-residue variance share, +-3 one-hot model (fit discovery, test
                 here), and for 0,789 the S/T neighbourhood (correlation, neighbours -> A, order vs disorder)
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.case_units import AA, Model  # noqa: E402
from scripts.pair_balance import burial3_of, proteins  # noqa: E402
from scripts.synthetic_columns import state5  # noqa: E402

KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8,
          P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
KD_BINS = [(-99, 0.5), (0.5, 1.0), (1.0, 1.6), (1.6, 99)]
SHSP = {"HSPB1": "P04792", "CRYAB": "P02511", "CRYAA": "P02489", "HSPB6": "O14558", "HSPB8": "Q9UJY1", "IbpA": "P0C054",
        "IbpB": "P0C058", "Hsp16.3": "P9WMK1", "Hsp16.5": "Q57733", "HSP26": "P15992", "HSP42": "Q12329"}
CS = {"p23": "Q15185", "SGT1": "Q9Y2Z0", "NUDC": "Q9Y266", "CHORDC1": "Q9UHD1", "SBA1": "P28707", "NUDCD3": "Q8IVD9", "SGT1y": "Q08446"}


def kd_window(s, half=9):
    v = torch.tensor([KD.get(c, 0.0) for c in s], dtype=torch.float64)
    c = torch.cat([torch.zeros(1, dtype=torch.float64), v.cumsum(0)])
    i = torch.arange(len(s)); lo = (i - half).clamp(min=0); hi = (i + half + 1).clamp(max=len(s))
    return (c[hi] - c[lo]) / (hi - lo)


def spearman(x, y):
    rx = torch.argsort(torch.argsort(x)).double(); ry = torch.argsort(torch.argsort(y)).double()
    return float(torch.corrcoef(torch.stack([rx, ry]))[0, 1])


def eta2(v, groups):
    v = v.double(); m = v.mean(); tot = float(((v - m) ** 2).sum()); by = defaultdict(list)
    for x, g in zip(v.tolist(), groups):
        by[g].append(x)
    bet = sum(len(xs) * (sum(xs) / len(xs) - float(m)) ** 2 for xs in by.values())
    return bet / tot


def boot(v, B=2000):
    if not v:
        return None
    t = torch.tensor(v, dtype=torch.float64); g = torch.Generator().manual_seed(0)
    bs = torch.stack([t[torch.randint(len(t), (len(t),), generator=g)].mean() for _ in range(B)])
    return [round(float(t.mean()), 5), round(float(torch.quantile(bs, 0.025)), 5), round(float(torch.quantile(bs, 0.975)), 5)]


class Feats:
    def __init__(self, d):
        iv = d["iv"]; o = torch.argsort(iv[:, 0], stable=True); self.iv = iv[o]; self.V = d["vocab"]
        self.st = torch.searchsorted(self.iv[:, 0].contiguous(), torch.arange(len(d["acc"]) + 1, dtype=torch.int32))
        self.idx = {x: i for i, x in enumerate(d["acc"])}

    def segments(self, acc, name):
        si = self.idx[acc]; vid = self.V.index(name)
        return [(r[1], r[2]) for r in self.iv[self.st[si]:self.st[si + 1]].tolist() if r[3] == vid]

    def mask(self, acc, name, L):
        m = torch.zeros(L, dtype=torch.bool)
        for s, e in self.segments(acc, name):
            m[s:min(e, L)] = True
        return m


def unit_h(M, s, l, u):
    M.run(s[:1022]); return M.cap[("h", l)][1:-1, u].clone()


def load_set(a, need_desc=True):
    first = json.load(open(a.exclude_json)) if a.exclude_json else []
    if a.set == "discovery":
        return proteins(a, False, a.n, 2)
    return proteins(a, True, a.n, 1, exclude=first)


# ---------------------------------------------------------------- tm (5,1181)
def cmd_tm(a, M, d, F):
    l, u, thr = 5, 1181, a.thr
    prot = load_set(a); b3, _ = burial3_of(prot)
    H, K, C = [], [], []
    for j, p in enumerate(prot):
        s = p["seq"]; L = len(s); h = unit_h(M, s, l, u); kd = kd_window(s); st = state5(p)
        tm = F.mask(p["acc"], "ft:TRANSMEM", L); sp = F.mask(p["acc"], "ft:SIGNAL", L)
        for i in range(L):
            if tm[i]:
                c = "TM"
            elif sp[i]:
                c = "SP"
            elif st[i] == 1 and b3[j][i] == 2:
                c = "buried helix (not TM/SP)"
            elif st[i] == 1 and b3[j][i] == 0:
                c = "exposed helix"
            elif st[i] == 2 and b3[j][i] == 2:
                c = "buried strand"
            else:
                c = "other"
            C.append(c)
        H.append(h); K.append(kd)
    H = torch.cat(H); K = torch.cat(K)
    out = {"proteins": len(prot), "positions": len(C), "thr": thr, "base_rate": round(float((H >= thr).double().mean()), 5),
           "spearman(h, KD19)": round(spearman(H, K), 3), "by_category": {}, "by_category_within_KD19_bins": {}}
    cats = sorted(set(C))
    for c in cats:
        m = torch.tensor([x == c for x in C])
        out["by_category"][c] = {"n": int(m.sum()), "rate": round(float((H[m] >= thr).double().mean()), 4), "median_h": round(float(H[m].median()), 3)}
    for lo, hi in KD_BINS:
        kb = (K >= lo) & (K < hi); row = {}
        for c in cats:
            m = kb & torch.tensor([x == c for x in C])
            if int(m.sum()) >= 20:
                row[c] = {"n": int(m.sum()), "rate": round(float((H[m] >= thr).double().mean()), 4), "median_h": round(float(H[m].median()), 3)}
        out["by_category_within_KD19_bins"][f"[{lo}, {hi})"] = row
    # context: distance to the nearest TM segment, and protein-level membership
    grp = defaultdict(list); bh = defaultdict(list); pmean = []
    for j, p in enumerate(prot):
        s = p["seq"]; L = len(s); h = unit_h(M, s, l, u); tm = F.mask(p["acc"], "ft:TRANSMEM", L); st = state5(p); has = bool(tm.any())
        if has:
            ix = torch.nonzero(tm).flatten(); dist = (torch.arange(L)[:, None] - ix[None, :]).abs().min(1).values
        for i in range(L):
            g = "no-TM protein" if not has else "TM segment" if tm[i] else "TM protein, <= 10 from a TM" if dist[i] <= 10 else "TM protein, > 10 from a TM"
            grp[g].append(float(h[i]))
            if st[i] == 1 and b3[j][i] == 2 and not tm[i]:
                bh["TM protein" if has else "no-TM protein"].append(float(h[i]))
        pmean.append((has, float(h.mean())))
    summ = lambda v: {"n": len(v), "median_h": round(float(torch.tensor(v).median()), 3), "rate": round(sum(x >= thr for x in v) / len(v), 4)} if v else None  # noqa: E731
    out["by distance to TM"] = {g: summ(v) for g, v in grp.items()}
    out["buried helix outside TM, by protein"] = {g: summ(v) for g, v in bh.items()}
    pp = torch.tensor([m for h_, m in pmean if h_]); nn_ = torch.tensor([m for h_, m in pmean if not h_])
    out["AUROC protein mean h (TM protein vs not)"] = round(float((pp[:, None] > nn_[None, :]).double().mean()), 3) if len(pp) and len(nn_) else None
    # interventions: TM segments inserted into soluble hosts; TM segments shuffled in place
    rng = random.Random(1)
    hosts = [p for p in prot if not F.segments(p["acc"], "ft:TRANSMEM") and not F.segments(p["acc"], "ft:SIGNAL") and 150 <= len(p["seq"]) <= 500]
    nat, ins, shf, hb = [], [], [], []
    for p in prot:
        if len(nat) >= 40:
            break
        segs = [(s0, e0) for s0, e0 in F.segments(p["acc"], "ft:TRANSMEM") if 15 <= e0 - s0 <= 30 and s0 >= 5 and e0 <= len(p["seq"]) - 5]
        if not segs:
            continue
        s0, e0 = segs[0]; s = p["seq"]; seg = s[s0:e0]
        nat.append(float(unit_h(M, s, l, u)[s0:e0].median()))
        v = []
        for hp in rng.sample(hosts, 3):
            t_ = hp["seq"]; m = len(t_) // 2
            v.append(float(unit_h(M, t_[:m] + seg + t_[m:], l, u)[m:m + len(seg)].median())); hb.append(float(unit_h(M, t_, l, u).median()))
        ins.append(sum(v) / 3)
        shf.append(sum(float(unit_h(M, s[:s0] + shuffled(seg, rng) + s[e0:], l, u)[s0:e0].median()) for _ in range(3)) / 3)
    md = lambda v: round(float(torch.tensor(v).median()), 3) if v else None  # noqa: E731
    out["TM segments (n, median of per-segment medians)"] = {"n": len(nat), "native": md(nat), "inserted into 3 soluble hosts": md(ins),
                                                              "shuffled in place": md(shf), "soluble hosts' own median": md(hb)}
    if nat:
        base = md(hb)
        out["TM retention (x - host) / (native - host), median"] = {
            "inserted": md([(x - base) / (y - base) for x, y in zip(ins, nat) if abs(y - base) > 0.1]),
            "shuffled in place": md([(x - base) / (y - base) for x, y in zip(shf, nat) if abs(y - base) > 0.1])}
    return out


# ---------------------------------------------------------------- shsp (3,1238)
def mmseqs(queries, targets, tmp, min_tcov=0.4):
    qf, tf, of = Path(tmp) / "q.fa", Path(tmp) / "t.fa", Path(tmp) / "o.m8"
    qf.write_text("".join(f">{k}\n{v}\n" for k, v in queries.items())); tf.write_text("".join(f">{k}\n{v}\n" for k, v in targets.items()))
    subprocess.run(["mmseqs", "easy-search", str(qf), str(tf), str(of), str(Path(tmp) / "w"), "-e", "1e-3", "-s", "7.5",
                    "--format-output", "query,target,fident,evalue,qcov,tcov", "--threads", "2", "-v", "1"], check=True)
    best = {}
    for line in open(of):
        q, t, fid, ev, qc, tc = line.strip().split("\t")[:6]
        if float(tc) < min_tcov:                       # the alignment must cover >= min_tcov of the seed
            continue
        if q not in best or float(ev) < best[q][2]:
            best[q] = (t, float(fid), float(ev), float(tc))
    return best


def frac(h, thr):
    return float((h >= thr).double().mean())


def shuffled(s, rng):
    x = list(s); rng.shuffle(x); return "".join(x)


def cmd_shsp(a, M, d, F):
    l, u, thr = 3, 1238, a.thr
    held = {x.split("\t")[0] for x in open(a.held) if not x.startswith("Entry")}
    seeds = {**{f"S|{k}": d["seq"][F.idx[v]] for k, v in SHSP.items()}, **{f"C|{k}": d["seq"][F.idx[v]] for k, v in CS.items()}}
    seed_acc = set(SHSP.values()) | set(CS.values())
    rng0 = random.Random(2); cand = [i for i, x in enumerate(d["acc"]) if x not in held and x not in seed_acc]
    smoke_pool = rng0.sample(cand, a.n_family)
    if a.set == "fresh":
        # family and CS members: every Swiss-Prot protein never inspected for this unit (not in its atlas top-64 or
        # reservoirs, not among the 600 discovery proteins, not in the discovery smoke's pool, not a seed);
        # non-family: held-out proteins with no hit (only one held-out protein is an sHSP)
        from atlas.core import SEQ_STRIDE
        at = torch.load(a.atlas, map_location="cpu", weights_only=False)
        toks = at["top_tok"][l, 0, u].tolist() + at["res_tok"][l, 0, u].flatten().tolist()
        seen = {d["acc"][t_ // SEQ_STRIDE] for t_ in toks if t_ >= 0} | {p["acc"] for p in proteins(a, False, 600, 2)}
        seen |= {d["acc"][i] for i in smoke_pool} | seed_acc
        allQ = {x: d["seq"][i][:1022] for i, x in enumerate(d["acc"]) if x not in seen}
        with tempfile.TemporaryDirectory(dir=a.tmp) as tmp:
            best_s = mmseqs(allQ, {k: v for k, v in seeds.items() if k.startswith("S|")}, tmp)
        with tempfile.TemporaryDirectory(dir=a.tmp) as tmp:
            best_c = mmseqs(allQ, {k: v for k, v in seeds.items() if k.startswith("C|")}, tmp)
        fam = set(best_s); cs = {q for q in best_c if q not in fam}
        Q = {k: allQ[k] for k in fam | cs}
        Q.update({x: d["seq"][i][:1022] for i, x in enumerate(d["acc"]) if x in held and x not in fam and x not in cs})
        out0 = {"excluded as inspected": len(seen)}
    else:
        Q = {d["acc"][i]: d["seq"][i][:1022] for i in smoke_pool}
        with tempfile.TemporaryDirectory(dir=a.tmp) as tmp:
            best_s = mmseqs(Q, {k: v for k, v in seeds.items() if k.startswith("S|")}, tmp)
        with tempfile.TemporaryDirectory(dir=a.tmp) as tmp:
            best_c = mmseqs(Q, {k: v for k, v in seeds.items() if k.startswith("C|")}, tmp)
        fam = {q for q in best_s}; cs = {q for q in best_c if q not in fam}
        out0 = {}
    fr, mx = {}, {}
    for k, s in Q.items():
        h = unit_h(M, s, l, u); fr[k] = frac(h, thr); mx[k] = float(h.max())
    non = [k for k in Q if k not in fam and k not in cs]

    def auroc(pos, neg):
        p = torch.tensor([fr[k] for k in pos]); n = torch.tensor([fr[k] for k in neg])
        return float(((p[:, None] > n[None, :]).double() + 0.5 * (p[:, None] == n[None, :]).double()).mean())
    med = lambda ks: round(float(torch.tensor([fr[k] for k in ks]).median()), 4) if ks else None  # noqa: E731
    q95 = lambda ks: round(float(torch.quantile(torch.tensor([fr[k] for k in ks]), 0.95)), 4) if ks else None  # noqa: E731
    low = [k for k in fam if best_s[k][1] < 0.35]
    out = {**out0, "proteins": len(Q), "family (hit to a canonical sHSP, e <= 1e-3, seed coverage >= 0.4)": len(fam), "CS-domain only": len(cs), "thr": thr,
           "family: median fraction above thr": med(fam), "non-family: median / q95 fraction": [med(non), q95(non)],
           "AUROC fraction (family vs non-family)": round(auroc(list(fam), non), 4) if fam else None,
           "family with best identity < 0.35": len(low), "low-identity family: median fraction": med(low),
           "CS-domain: median / max fraction": [med(list(cs)), round(max((fr[k] for k in cs), default=0), 4)],
           "family members (acc, identity, fraction, max)": sorted(((k, round(best_s[k][1], 2), round(fr[k], 3), round(mx[k], 2)) for k in fam), key=lambda x: -x[2]),
           "non-family with fraction >= 0.05": sorted(((k, round(fr[k], 3)) for k in non if fr[k] >= 0.05), key=lambda x: -x[1])[:20]}
    # perturbations on family members (up to 30, longest first so both parts have room)
    rng = random.Random(1)
    fm = sorted(fam, key=lambda k: -len(Q[k]))[:30] if a.set == "discovery" else rng.sample(sorted(k for k in fam if len(Q[k]) >= 120), min(30, sum(len(Q[k]) >= 120 for k in fam)))
    hosts = [k for k in non if 150 <= len(Q[k]) <= 500]
    bro, trans, shuf = [], [], []
    for k in fm:
        s = Q[k]; L = len(s); c = int(0.4 * L); h0 = unit_h(M, s, l, u)
        n0, c0 = frac(h0[:c], thr), frac(h0[c:], thr)
        nC = [frac(unit_h(M, s[:c] + shuffled(s[c:], rng), l, u)[:c], thr) for _ in range(3)]
        cN = [frac(unit_h(M, shuffled(s[:c], rng) + s[c:], l, u)[c:], thr) for _ in range(3)]
        bro.append({"acc": k, "N40 native": round(n0, 3), "N40 with C60 shuffled": round(sum(nC) / 3, 3), "C60 native": round(c0, 3), "C60 with N40 shuffled": round(sum(cN) / 3, 3)})
        i = int(torch.argmax(h0)); i = min(max(i, 10), L - 11); w = s[i - 10:i + 11]
        for hk in rng.sample(hosts, 5):
            t = Q[hk]; p = rng.randint(15, len(t) - 36); new = t[:p] + w + t[p + 21:]
            trans.append(float(unit_h(M, new, l, u)[p + 10]))
        shuf += [frac(unit_h(M, shuffled(s, rng), l, u), thr) for _ in range(3)]

    def rel(xs, a_, b_):
        r = [x[b_] / x[a_] for x in xs if x[a_] > 0]
        return round(float(torch.tensor(r).median()), 3) if r else None
    out["broadcast"] = {"per protein": bro, "median N40(C60 shuffled) / N40 native": rel(bro, "N40 native", "N40 with C60 shuffled"),
                        "median C60(N40 shuffled) / C60 native": rel(bro, "C60 native", "C60 with N40 shuffled")}
    out["window transplants (+-10 around the family member's maximum, 5 hosts each)"] = {
        "n": len(trans), "share >= thr": round(sum(v >= thr for v in trans) / max(len(trans), 1), 3),
        "median value": round(float(torch.tensor(trans).median()), 3) if trans else None}
    lo_i = [k for k in fam if best_s[k][1] < 0.35]; hi_i = [k for k in fam if best_s[k][1] >= 0.5]
    out["family by best identity to a seed: median fraction (n)"] = {"< 0.35": [med(lo_i), len(lo_i)], "0.35-0.5": [med([k for k in fam if 0.35 <= best_s[k][1] < 0.5]), sum(0.35 <= best_s[k][1] < 0.5 for k in fam)], ">= 0.5": [med(hi_i), len(hi_i)]}
    out["composition shuffles (3 per member)"] = {"n": len(shuf), "median fraction": round(float(torch.tensor(shuf).median()), 4) if shuf else None,
                                                   "max fraction": round(max(shuf), 4) if shuf else None}
    return out


# ---------------------------------------------------------------- gc (5,668)
GARP, FYMINK = set("GARP"), set("FYMINK")


def gc_index(s):
    return (sum(c in GARP for c in s) - sum(c in FYMINK for c in s)) / len(s)


def cmd_gc(a, M, d, F):
    l, u = 5, 668
    prot = load_set(a)
    mean, idx, allh, pid, cen = [], [], [], [], []
    for j, p in enumerate(prot):
        h = unit_h(M, p["seq"], l, u); mean.append(float(h.mean())); idx.append(gc_index(p["seq"]))
        allh.append(h); pid += [j] * len(h); cen += list(p["seq"][:len(h)])
    allh = torch.cat(allh); mu = float(torch.tensor(mean).mean())
    out = {"proteins": len(prot), "spearman(protein mean h, GARP - FYMINK)": round(spearman(torch.tensor(mean), torch.tensor(idx)), 3),
           "eta2 protein (share of per-position variance between proteins)": round(eta2(allh, pid), 3),
           "eta2 centre residue": round(eta2(allh, cen), 3), "mean of protein means": round(mu, 3)}
    rng = random.Random(1)
    top = sorted(range(len(prot)), key=lambda j: -mean[j])[:20]
    ret = []
    for j in top:
        sm = sum(float(unit_h(M, shuffled(prot[j]["seq"], rng), l, u).mean()) for _ in range(3)) / 3
        ret.append((sm - mu) / (mean[j] - mu))
    out["shuffles of the top-20 proteins: retention (shuffled - mu) / (native - mu), median"] = round(float(torch.tensor(ret).median()), 3)
    long_ = [j for j in range(len(prot)) if len(prot[j]["seq"]) >= 150]
    hi = sorted(long_, key=lambda j: -idx[j])[:20]; lo = sorted(long_, key=lambda j: idx[j])[:20]
    shifts = []
    for jh, jl in zip(hi, lo):
        A, B = prot[jh]["seq"][:150], prot[jl]["seq"][:150]
        mA, mB = float(unit_h(M, A, l, u).mean()), float(unit_h(M, B, l, u).mean())
        if abs(mA - mB) < 0.1:
            continue
        hab = unit_h(M, A + B, l, u); hba = unit_h(M, B + A, l, u)
        shifts.append((float(hab[:150].mean()) - mA) / (mB - mA)); shifts.append((float(hba[150:].mean()) - mA) / (mB - mA))
        shifts.append((float(hab[150:].mean()) - mB) / (mA - mB)); shifts.append((float(hba[:150].mean()) - mB) / (mA - mB))
    out["chimeras (high-index 150 + low-index 150, both orders): shift of each half toward its partner, as a share of the gap"] = {
        "n": len(shifts), "median": round(float(torch.tensor(shifts).median()), 3) if shifts else None,
        "q25 / q75": [round(float(torch.quantile(torch.tensor(shifts), q)), 3) for q in (0.25, 0.75)] if shifts else None}
    return out


# ---------------------------------------------------------------- nterm (1,746)
def cmd_nterm(a, M, d, F):
    l, u, thr = 1, 746, a.thr
    prot = load_set(a)
    H, K, C = [], [], []
    for p in prot:
        s = p["seq"]; L = len(s); h = unit_h(M, s, l, u); kd = kd_window(s)
        sp = F.mask(p["acc"], "ft:SIGNAL", L); pp = F.mask(p["acc"], "ft:PROPEP", L)
        ntm, itm = torch.zeros(L, dtype=torch.bool), torch.zeros(L, dtype=torch.bool)
        for s0, e0 in F.segments(p["acc"], "ft:TRANSMEM"):
            (ntm if s0 < 60 else itm)[s0:min(e0, L)] = True
        for i in range(L):
            C.append("SIGNAL" if sp[i] else "TM starting < 60" if ntm[i] else "TM starting >= 60" if itm[i] else "PROPEP" if pp[i]
                     else "first 30 (other)" if i < 30 else "other")
        H.append(h); K.append(kd)
    H = torch.cat(H); K = torch.cat(K); cats = sorted(set(C))
    out = {"proteins": len(prot), "positions": len(C), "thr": thr, "base_rate": round(float((H >= thr).double().mean()), 5),
           "spearman(h, KD19)": round(spearman(H, K), 3), "by_category": {}, "by_category_within_KD19_bins": {}}
    for c in cats:
        m = torch.tensor([x == c for x in C])
        out["by_category"][c] = {"n": int(m.sum()), "rate": round(float((H[m] >= thr).double().mean()), 4), "median_h": round(float(H[m].median()), 3)}
    for lo, hi in KD_BINS:
        kb = (K >= lo) & (K < hi); row = {}
        for c in cats:
            m = kb & torch.tensor([x == c for x in C])
            if int(m.sum()) >= 20:
                row[c] = {"n": int(m.sum()), "rate": round(float((H[m] >= thr).double().mean()), 4), "median_h": round(float(H[m].median()), 3)}
        out["by_category_within_KD19_bins"][f"[{lo}, {hi})"] = row
    # moves
    mv_sp, mv_tm = [], []
    for p in prot:
        s = p["seq"]
        for s0, e0 in F.segments(p["acc"], "ft:SIGNAL"):
            if s0 == 0 and 12 <= e0 <= 45 and len(s) - e0 >= 100 and len(mv_sp) < 40:
                rest = s[e0:]; mid = len(rest) // 2; new = rest[:mid] + s[:e0] + rest[mid:]
                n0 = frac(unit_h(M, s, l, u)[:e0], thr); n1 = frac(unit_h(M, new, l, u)[mid:mid + e0], thr)
                mv_sp.append((round(n0, 3), round(n1, 3)))
        for s0, e0 in F.segments(p["acc"], "ft:TRANSMEM"):
            if s0 >= 60 and 15 <= e0 - s0 <= 30 and len(mv_tm) < 40:
                seg = s[s0:e0]; new = "M" + seg + s[:s0] + s[e0:]
                n0 = frac(unit_h(M, s, l, u)[s0:e0], thr); n1 = frac(unit_h(M, new, l, u)[1:1 + len(seg)], thr)
                mv_tm.append((round(n0, 3), round(n1, 3)))
                break
    mean2 = lambda xs, k: round(sum(x[k] for x in xs) / max(len(xs), 1), 3)  # noqa: E731
    out["signal peptides moved to the middle"] = {"n": len(mv_sp), "share above thr native / moved": [mean2(mv_sp, 0), mean2(mv_sp, 1)]}
    out["internal TM segments moved to the N-terminus"] = {"n": len(mv_tm), "share above thr native / moved": [mean2(mv_tm, 0), mean2(mv_tm, 1)]}
    # readers: silence the unit (post-GELU 0, no mask) at top positions; change in layer-2 pre-activations
    R = [1006, 455]; du, dc = {r: [] for r in R}, {r: [] for r in R}; n = 0
    wout = M.layers[l].output.dense.weight.norm(dim=0)
    for p in prot:
        s = p["seq"][:1022]; h = unit_h(M, s, l, u)
        top = torch.nonzero(h >= thr).flatten().tolist()[:3]
        for i in top:
            if n >= 150:
                break
            M.run(s); g = M.cap[("g", l)][i + 1]; base = M.cap[("h", l + 1)][i + 1, R].clone()
            wm = g.abs() * wout; ctl = next(v for v in torch.argsort((wm - wm[u]).abs()).tolist() if v != u)
            M.run(s, (l, i, u)); au = M.cap[("h", l + 1)][i + 1, R] - base
            M.run(s, (l, i, ctl)); ac = M.cap[("h", l + 1)][i + 1, R] - base
            for k, r in enumerate(R):
                du[r].append(float(au[k])); dc[r].append(float(ac[k]))
            n += 1
    out["readers: change in layer-2 pre-activation at the position when the unit is silenced (mean, 95% CI)"] = {
        f"2/{r}": {"unit": boot(du[r]), "write-matched control": boot(dc[r])} for r in R}
    out["readers: positions"] = n
    # downstream: silence the unit at residues 1..(SP end + 10) in a forward with 15% of SP positions masked,
    # against a control unit with the closest summed |GELU| x |w_out| over the same positions in the same forward
    rng = random.Random(1); du, dc, nP = [], [], 0
    for p in prot:
        s = p["seq"][:1022]
        sp = [(s0, e0) for s0, e0 in F.segments(p["acc"], "ft:SIGNAL") if s0 == 0 and 12 <= e0 <= 45]
        if not sp or len(s) < 100 or nP >= 60:
            continue
        e0 = sp[0][1]; k = max(2, round(0.15 * e0)); mpos = sorted(rng.sample(range(1, e0), k)); reg = list(range(0, min(e0 + 10, len(s))))
        ids = M.tok(s, return_tensors="pt")["input_ids"]; x = ids.clone()
        for i in mpos:
            x[0, i + 1] = M.tok.mask_token_id
        lp0 = M.run(x); g = M.cap[("g", l)][[i + 1 for i in reg]]
        wm = (g.abs() * wout[None, :]).sum(0); ctl = next(v for v in torch.argsort((wm - wm[u]).abs()).tolist() if v != u)
        tr = [AA.find(s[i]) for i in mpos]
        for unit, acc in ((u, du), (ctl, dc)):
            hk = M.layers[l].intermediate.register_forward_hook(lambda _m, _i, o, unit=unit: o.index_put((torch.zeros(len(reg), dtype=torch.long), torch.tensor(reg) + 1, torch.full((len(reg),), unit)), torch.zeros(len(reg))))
            lp = M.run(x); hk.remove()
            acc.append(sum(float(lp[i + 1, t_] - lp0[i + 1, t_]) for i, t_ in zip(mpos, tr) if t_ >= 0) / len(mpos))
        nP += 1
    out["downstream: silencing over the SP (+10), change in mean log p(true) at masked SP positions (mean, 95% CI)"] = {
        "proteins": nP, "unit": boot(du), "write-matched control": boot(dc)}
    return out


# ---------------------------------------------------------------- tok (0,767 / 0,789)
def onehot(s, i):
    x = torch.zeros(6 * 21)
    for k, o in enumerate((-3, -2, -1, 1, 2, 3)):
        j = i + o; c = s[j] if 0 <= j < len(s) else None
        x[k * 21 + (AA.index(c) if c is not None and c in AA else 20)] = 1.0
    return x


def cmd_tok(a, M, d, F):
    l, u = map(int, a.unit.split(",")); thr = a.thr
    prot = load_set(a)
    allh, cen = [], []
    for p in prot:
        h = unit_h(M, p["seq"], l, u); allh.append(h); cen += list(p["seq"][:len(h)])
    H = torch.cat(allh)
    out = {"proteins": len(prot), "thr": thr, "eta2 centre residue": round(eta2(H, cen), 3)}
    rate = Counter(); n = Counter()
    for v, c in zip(H.tolist(), cen):
        n[c] += 1; rate[c] += v >= thr
    out["rate above thr by centre residue (top 6)"] = sorted(((c, n[c], round(rate[c] / n[c], 4)) for c in n if n[c] >= 20), key=lambda x: -x[2])[:6]
    out["X tokens: n, rate"] = [n["X"], round(rate["X"] / n["X"], 4) if n["X"] else None]
    # +-3 one-hot ridge within the main centre residues, fit on discovery proteins, tested here
    disc = proteins(a, False, a.n, 2) if a.set == "fresh" else None
    centres = a.centres.split(",")
    for c0 in centres:
        def rows(ps, hs):
            X, Y = [], []
            for p, h in zip(ps, hs):
                for i, c in enumerate(p["seq"][:len(h)]):
                    if c == c0:
                        X.append(onehot(p["seq"], i)); Y.append(float(h[i]))
            return torch.stack(X).double(), torch.tensor(Y, dtype=torch.float64)
        Xt, Yt = rows(prot, allh)
        if disc is not None:
            Xd, Yd = rows(disc, [unit_h(M, p["seq"], l, u) for p in disc])
        else:
            Xd, Yd = Xt, Yt
        Xd1 = torch.cat([Xd, torch.ones(len(Xd), 1, dtype=torch.float64)], 1)
        w = torch.linalg.solve(Xd1.T @ Xd1 + 1.0 * torch.eye(Xd1.shape[1], dtype=torch.float64), Xd1.T @ Yd)
        pred = torch.cat([Xt, torch.ones(len(Xt), 1, dtype=torch.float64)], 1) @ w
        r2 = 1 - float(((Yt - pred) ** 2).sum() / ((Yt - Yt.mean()) ** 2).sum())
        top = torch.topk(w[:-1], 6); bot = torch.topk(-w[:-1], 4)
        nm = lambda k: f"{(-3, -2, -1, 1, 2, 3)[k // 21]:+d}{(AA + '*')[k % 21]}"  # noqa: E731
        out[f"centre {c0}: +-3 ridge (fit on discovery), R^2 here"] = {"n": len(Yt), "r2": round(r2, 3),
            "largest weights": [(nm(int(k)), round(float(v), 3)) for v, k in zip(top.values, top.indices)],
            "most negative": [(nm(int(k)), round(-float(v), 3)) for v, k in zip(bot.values, bot.indices)]}
    if (l, u) == (0, 789):
        ST = set("ST"); b3, _ = burial3_of(prot)
        for c0 in ("C", "S"):
            v, k = [], []
            for p, h in zip(prot, allh):
                s = p["seq"]
                for i, c in enumerate(s[:len(h)]):
                    if c == c0:
                        v.append(float(h[i])); k.append(sum(s[j] in ST for j in range(max(0, i - 3), min(len(s), i + 4)) if j != i))
            out[f"centre {c0}: spearman(h, S/T count in +-3)"] = round(spearman(torch.tensor(v), torch.tensor(k, dtype=torch.float64)), 3) if v else None
        dis, ordd, mut = [], [], []
        for j, (p, h) in enumerate(zip(prot, allh)):
            s = p["seq"]; st = state5(p)
            for i, c in enumerate(s[:len(h)]):
                if c != "C" or i < 3 or i >= len(s) - 3:
                    continue
                nst = sum(s[x] in ST for x in range(i - 3, i + 4) if x != i)
                if nst >= 4:
                    (dis if int(st[i]) == 3 else ordd if int(st[i]) in (0, 1, 2) else []).append(float(h[i]))
                if float(h[i]) >= thr and len(mut) < 60:
                    new = s[:i - 3] + "AAA" + "C" + "AAA" + s[i + 4:]
                    mut.append(float(unit_h(M, new, l, u)[i]))
        md = lambda xs: round(float(torch.tensor(xs).median()), 3) if xs else None  # noqa: E731
        out["C with >= 4 S/T among +-3: median h disordered / ordered (n)"] = [md(dis), md(ordd), len(dis), len(ordd)]
        out["C above thr with +-3 -> A: share still above thr, median h"] = [round(sum(x >= thr for x in mut) / max(len(mut), 1), 3), md(mut), len(mut)]
        lone, clus, rise = [], [], []
        for p, h in zip(prot, allh):
            s = p["seq"]
            for i, c in enumerate(s[:len(h)]):
                if c != "C":
                    continue
                nb = [x for x in range(max(0, i - 3), min(len(s), i + 4)) if x != i and s[x] == "C"]
                (clus if nb else lone).append(float(h[i]))
                if nb and len(rise) < 80:
                    new = "".join("A" if x in nb else ch for x, ch in enumerate(s))
                    rise.append(float(unit_h(M, new, l, u)[i]) - float(h[i]))
        out["C: median h with no other C in +-3 / with >= 1 (n)"] = [md(lone), md(clus), len(lone), len(clus)]
        out["C with a C neighbour: neighbouring C -> A, median change in h (n)"] = [md(rise), len(rise)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["tm", "shsp", "gc", "nterm", "tok"])
    ap.add_argument("--set", choices=["discovery", "fresh"], required=True)
    ap.add_argument("--unit", default=None); ap.add_argument("--thr", type=float, default=None); ap.add_argument("--centres", default="P")
    ap.add_argument("--data", required=True); ap.add_argument("--desc-dir", required=True); ap.add_argument("--uniref", required=True)
    ap.add_argument("--held", default="lists/heldout_created_after_2021-11-17.tsv"); ap.add_argument("--exclude-json", default=None)
    ap.add_argument("--n", type=int, default=600); ap.add_argument("--n-family", type=int, default=10010)
    ap.add_argument("--tmp", default="/tmp"); ap.add_argument("--out", required=True); ap.add_argument("--atlas", default=None)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    d = torch.load(a.data, weights_only=False); F = Feats(d); M = Model()
    res = {"test": a.test, "set": a.set, **globals()[f"cmd_{a.test}"](a, M, d, F)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    name = f"{a.test}{'_' + a.unit.replace(',', '_') if a.unit else ''}_{a.set}.json"
    (Path(a.out) / name).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main()
