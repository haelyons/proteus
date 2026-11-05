"""Ruler v1: is a column's sample associated with the descriptor vocabulary, beyond composition and sample size?

It reads only an atlas result (reservoirs per bin, top-K) and per-residue descriptor tables; it never sees
what a column is (unit, latent, synthetic). Calls follow results/plan_ruler_v1.md section 7:
  - samples: the reservoir of a bin, de-duplicated to one token per UniRef50 cluster (smallest key, so the
    choice stays uniform);
  - descriptors (categorical): state5 (loop / helix / strand at pLDDT >= 70, disordered < 50, uncertain
    50-70); burial3 (terciles of the C-beta count, pLDDT >= 70); rama4 (alpha-R / beta / alpha-L / other,
    pLDDT >= 70); residue identity (20);
  - composition-matched null: each sampled residue a contributes P(x | a), estimated on the background pool,
    to the expected distribution; Monte Carlo draws from those per-residue distributions give p-values,
    per descriptor (G statistic) and per state (count, one-sided);
  - residue identity is scored against the background composition (multinomial Monte Carlo);
  - calls at the peak (q99.999 bin): structural (primary structural descriptor significant) / composition
    (residue-specific: effective number of residues <= 4, no structural excess) / context (both) / none;
    modes = enriched states of the primary descriptor (p < 0.01 and log2 ratio >= 0.5); several modes are
    reported as such, never as several jobs;
  - persistence (descriptive): the peak's top mode at q99 and q90 (95% interval above 0 and at least half
    the peak's log2 ratio).
"""
from __future__ import annotations

import math

import torch

AA = "ACDEFGHIKLMNPQRSTVWY"
DESC = {"state5": 5, "burial3": 3, "rama4": 4}
LEVELS = {"q90": 1, "q99": 2, "q99.9": 3, "q99.99": 4, "peak": 5}


class Tables:
    """Per-residue descriptor tables for a set of proteins (dicts as in scripts/synthetic_columns.load_subset)."""

    def __init__(self, prot):
        rows, self.row_of = 0, {}
        aa, st, bu, ra, cl, plddt = [], [], [], [], [], []
        names = {}
        for p in prot:
            n = len(p["seq"])
            self.row_of[p["seq_idx"]] = rows
            rows += n
            aa.append(torch.tensor([AA.find(c) for c in p["seq"]]))
            s = p["ss3"].long().clone(); s[p["plddt"] < 70] = 4; s[p["plddt"] < 50] = 3
            st.append(s)
            bu.append(torch.where(p["plddt"] >= 70, p["burial"], torch.full((n,), float("nan"))))
            phi, psi = p["phi"], p["psi"]
            r = torch.full((n,), -1, dtype=torch.long)
            ok = (p["plddt"] >= 70) & ~torch.isnan(phi) & ~torch.isnan(psi)
            r[ok & (phi > 0)] = 2
            r[ok & (phi <= 0) & (psi >= -100) & (psi <= 45)] = 0
            r[ok & (phi <= 0) & ((psi > 45) | (psi < -100))] = 1
            ra.append(r)
            cl.append(torch.full((n,), names.setdefault(p["cluster"], len(names)), dtype=torch.long))
            plddt.append(p["plddt"])
        self.aa, self.state5, self.rama4, self.cluster = torch.cat(aa), torch.cat(st), torch.cat(ra), torch.cat(cl)
        b = torch.cat(bu)
        cut = torch.quantile(b[~torch.isnan(b)], torch.tensor([1 / 3, 2 / 3]))
        self.burial3 = torch.full_like(self.aa, -1)
        okb = ~torch.isnan(b)
        self.burial3[okb] = torch.bucketize(b[okb], cut, right=True)  # 0 exposed, 1 middle, 2 buried
        self.n = rows
        self.desc = {"state5": self.state5, "burial3": self.burial3, "rama4": self.rama4}
        self.p_x_given_a = {}
        for k, x in self.desc.items():
            m = (x >= 0) & (self.aa >= 0)
            cnt = torch.zeros(20, DESC[k])
            cnt.index_put_((self.aa[m], x[m]), torch.ones(int(m.sum())), accumulate=True)
            self.p_x_given_a[k] = (cnt + 0.5) / (cnt + 0.5).sum(1, keepdim=True)
        bg = torch.bincount(self.aa[self.aa >= 0], minlength=20).float()
        self.p_aa = bg / bg.sum()

    def rows(self, tok: torch.Tensor) -> torch.Tensor:
        """packed token ids -> table rows (-1 if the protein is not in the table)."""
        out = torch.full_like(tok, -1)
        for i, t in enumerate(tok.tolist()):
            if t < 0:
                continue
            base = self.row_of.get(t // 1024)
            if base is not None:
                out[i] = base + t % 1024
        return out


def dedup(rows: torch.Tensor, cluster: torch.Tensor) -> torch.Tensor:
    seen, keep = set(), []
    for r in rows.tolist():
        if r < 0:
            continue
        c = int(cluster[r])
        if c not in seen:
            seen.add(c); keep.append(r)
    return torch.tensor(keep, dtype=torch.long)


def score_desc(T: Tables, rows, k, B, gen):
    x, a = T.desc[k][rows], T.aa[rows]
    ok = (x >= 0) & (a >= 0)
    x, a = x[ok], a[ok]
    n = int(x.numel())
    if n < 10:
        return None
    P = T.p_x_given_a[k][a]                                   # [n, K] per-sample null distribution
    K = P.shape[1]
    obs = torch.bincount(x, minlength=K).float()
    exp = P.sum(0)
    draws = torch.multinomial(P, B, replacement=True, generator=gen)  # [n, B]
    sim = torch.zeros(B, K)
    sim.scatter_add_(1, draws.T, torch.ones(B, n))
    G = lambda c: (c * torch.log((c + 1e-9) / exp)).sum(-1)  # noqa: E731
    p = float(((G(sim) >= G(obs)).sum() + 1) / (B + 1))
    lr = torch.log2((obs + 0.5) / (exp + 0.5))
    ps = ((sim >= obs).sum(0).float() + 1) / (B + 1)
    q = obs / n; se = torch.sqrt(q * (1 - q) / n + 1e-12); ef = exp / n
    lo = torch.log2(torch.clamp(q - 1.96 * se, min=1e-6) / ef); hi = torch.log2(torch.clamp(q + 1.96 * se, min=1e-6) / ef)
    return {"n": n, "p": p, "log2": lr.tolist(), "p_state": ps.tolist(), "ci_lo": lo.tolist(), "ci_hi": hi.tolist(),
            "obs": obs.tolist(), "exp": exp.tolist()}


def score_aa(T: Tables, rows, B, gen):
    a = T.aa[rows]; a = a[a >= 0]; n = int(a.numel())
    if n < 10:
        return None
    obs = torch.bincount(a, minlength=20).float()
    exp = T.p_aa * n
    sim = torch.multinomial(T.p_aa.expand(B, 20), n, replacement=True, generator=gen)
    simc = torch.zeros(B, 20).scatter_add_(1, sim, torch.ones(B, n))
    G = lambda c: (c * torch.log((c + 1e-9) / exp)).sum(-1)  # noqa: E731
    p = float(((G(simc) >= G(obs)).sum() + 1) / (B + 1))
    f = obs / n
    eff = float(torch.exp(-(f[f > 0] * torch.log(f[f > 0])).sum()))   # effective number of residues
    top = [(AA[i], round(float(f[i]), 3)) for i in f.argsort(descending=True)[:4].tolist()]
    return {"n": n, "p": p, "eff_residues": eff, "top": top}


def score_column(T: Tables, res_tok_col: torch.Tensor, B=2000, seed=0):
    """res_tok_col [n_bins, Kr] packed tokens (keys ascending) for one column and pole."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for name, b in LEVELS.items():
        rows = dedup(T.rows(res_tok_col[b]), T.cluster)
        lvl = {"n_samples": int((res_tok_col[b] >= 0).sum()), "n_clusters": int(rows.numel())}
        if rows.numel() >= 10:
            lvl["aa"] = score_aa(T, rows, B, gen)
            for k in DESC:
                lvl[k] = score_desc(T, rows, k, B, gen)
        out[name] = lvl
    return out


def calls(scores: list[dict], alpha=0.01):
    """Per-column calls from score_column outputs, with BH across columns x {structural, composition}."""
    tests = []
    for i, s in enumerate(scores):
        pk = s["peak"]
        if "aa" not in pk:
            continue
        cand = [(k, pk[k]) for k in DESC if pk.get(k)]
        if not cand:
            continue
        def jsd_excess(d):
            q = torch.tensor(d["obs"]) / max(d["n"], 1); e = torch.tensor(d["exp"]) / max(d["n"], 1)
            m = 0.5 * (q + e)
            kl = lambda u, v: float((u[u > 0] * torch.log2(u[u > 0] / v[u > 0])).sum())  # noqa: E731
            return 0.5 * kl(q, m) + 0.5 * kl(e, m)
        prim, d = max(cand, key=lambda kd: (-math.log10(kd[1]["p"]), jsd_excess(kd[1])))
        s["primary"] = prim
        s["primary_jsd"] = jsd_excess(d)
        tests.append((d["p"], i, "struct"))
        if pk["aa"]:
            tests.append((pk["aa"]["p"], i, "comp"))
    tests.sort()
    m = len(tests); passed = set()
    for r, (p, i, kind) in enumerate(tests, start=1):
        if p <= alpha * r / m:
            passed = {(j, kd) for (_, j, kd) in tests[:r]}
    out = []
    for i, s in enumerate(scores):
        pk = s["peak"]
        c = {"col": i, "n_clusters_peak": pk.get("n_clusters", 0)}
        if "aa" not in pk or "primary" not in s:
            c["call"] = "insufficient samples"; out.append(c); continue
        st = (i, "struct") in passed
        specific = pk["aa"] is not None and pk["aa"]["eff_residues"] <= 4.0 and (i, "comp") in passed
        d = pk[s["primary"]]
        modes = [x for x in range(len(d["log2"])) if d["p_state"][x] < 0.01 and d["log2"][x] >= 0.5] if st else []
        c.update({"primary": s["primary"], "modes": modes, "aa_top": pk["aa"]["top"], "eff_residues": round(pk["aa"]["eff_residues"], 2)})
        if st and specific:
            c["call"] = "context association"
        elif st:
            c["call"] = "structural association"
        elif specific:
            c["call"] = "composition association"
        else:
            c["call"] = "no association found by this procedure"
        if st and modes:
            c["mode_label"] = "one mode" if len(modes) == 1 else "several modes"
            top = max(modes, key=lambda x: d["log2"][x])
            peak_lr = d["log2"][top]
            pres = {}
            for lv in ("q99", "q90"):
                dl = s[lv].get(s["primary"])
                pres[lv] = bool(dl and dl["ci_lo"][top] > 0 and dl["log2"][top] >= 0.5 * peak_lr)
            c["persistence"] = "to q90" if pres["q99"] and pres["q90"] else ("to q99" if pres["q99"] else "peak only")
            c["top_mode"] = top
        out.append(c)
    return out
