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

from atlas.core import SEQ_STRIDE

AA = "ACDEFGHIKLMNPQRSTVWY"
DESC = {"state5": 5, "burial3": 3, "rama4": 4}
LEVELS = {"q90": 1, "q99": 2, "q99.9": 3, "q99.99": 4, "peak": 5}


class Tables:
    """Per-residue descriptor tables for a set of proteins (dicts as in scripts/synthetic_columns.load_subset)."""

    def __init__(self, prot):
        rows, self.row_of, self.len_of = 0, {}, {}
        aa, st, bu, ra, cl, plddt = [], [], [], [], [], []
        names = {}
        for p in prot:
            n = len(p["seq"])
            self.row_of[p["seq_idx"]] = rows
            self.len_of[p["seq_idx"]] = n
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
        bc = torch.full_like(self.aa, -1)                             # raw C-beta count, counts >= 45 pooled
        bc[okb] = b[okb].round().long().clamp(max=45)
        self.n = rows
        self.desc = {"state5": self.state5, "burial3": self.burial3, "rama4": self.rama4}
        self.p_x_given_a, self.p_a_given_x = {}, {}
        for k, x in self.desc.items():
            m = (x >= 0) & (self.aa >= 0)
            cnt = torch.zeros(20, DESC[k])
            cnt.index_put_((self.aa[m], x[m]), torch.ones(int(m.sum())), accumulate=True)
            self.p_x_given_a[k] = (cnt + 0.5) / (cnt + 0.5).sum(1, keepdim=True)
            self.p_a_given_x[k] = ((cnt + 0.5) / (cnt + 0.5).sum(0, keepdim=True)).T.contiguous()   # [K, 20]
        # v1.2: the residue test conditions on the finest value of the primary descriptor (burial: the count)
        self.fine = {"state5": self.state5, "burial3": bc, "rama4": self.rama4}
        self.p_a_given_fine = {}
        for k, x in self.fine.items():
            m = (x >= 0) & (self.aa >= 0)
            cnt = torch.zeros(int(x.max()) + 1, 20)
            cnt.index_put_((x[m], self.aa[m]), torch.ones(int(m.sum())), accumulate=True)
            self.p_a_given_fine[k] = (cnt + 0.5) / (cnt + 0.5).sum(1, keepdim=True)
        bg = torch.bincount(self.aa[self.aa >= 0], minlength=20).float()
        self.p_aa = bg / bg.sum()

    def rows(self, tok: torch.Tensor) -> torch.Tensor:
        """packed token ids -> table rows (-1 if the protein is not in the table)."""
        out = torch.full_like(tok, -1)
        for i, t in enumerate(tok.tolist()):
            if t < 0:
                continue
            q, pos = divmod(t, SEQ_STRIDE)
            base = self.row_of.get(q)
            if base is not None and pos < self.len_of[q]:
                out[i] = base + pos
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


def score_desc(T: Tables, rows, k, B, gen, gen_cond=None):
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
    out = {"n": n, "p": p, "log2": lr.tolist(), "p_state": ps.tolist(), "ci_lo": lo.tolist(), "ci_hi": hi.tolist(),
           "obs": obs.tolist(), "exp": exp.tolist()}
    # v1.1: second mode tested among the samples outside the top mode, against the same per-residue null
    # restricted to the other states (a two-mode sample splits between its modes, so each mode's marginal
    # log2 ratio is diluted by the other; the conditional test is not)
    cand = [s for s in range(K) if ps[s] < 0.01 and lr[s] >= 0.5]
    if cand:
        top = max(cand, key=lambda s: float(lr[s]))
        rest = x != top
        if gen_cond is not None and int(rest.sum()) >= 10:   # own generator: v1's draws are unchanged
            Pr = P[rest].clone(); Pr[:, top] = 0; Pr = Pr / Pr.sum(1, keepdim=True)
            nr = int(rest.sum()); obr = torch.bincount(x[rest], minlength=K).float(); exr = Pr.sum(0)
            dr = torch.multinomial(Pr, B, replacement=True, generator=gen_cond)
            sr = torch.zeros(B, K); sr.scatter_add_(1, dr.T, torch.ones(B, nr))
            out["cond"] = {"top": top, "n": nr, "log2": torch.log2((obr + 0.5) / (exr + 0.5)).tolist(),
                           "p_state": (((sr >= obr).sum(0).float() + 1) / (B + 1)).tolist()}
    return out


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
    pr = ((simc >= obs).sum(0).float() + 1) / (B + 1)                  # one-sided, per residue
    lr = torch.log2((obs + 0.5) / (exp + 0.5))
    return {"n": n, "p": p, "eff_residues": eff, "top": top, "p_res": pr.tolist(), "log2_res": lr.tolist()}


def score_aa_given(T: Tables, rows, k, B, gen, fine=False):
    """v1.1: residue identity against the residues expected given each sample's state on descriptor k
    (P(a | x), background composition where x is undefined): residue enrichment beyond the structure.
    v1.2 (fine=True): x is the descriptor's finest value (burial: the raw count, not the tercile), and the
    test statistic is the largest standardised residue excess (p_max), not the 19-df G (a single residue's
    excess is diluted by the other 19)."""
    a, x = T.aa[rows], (T.fine if fine else T.desc)[k][rows]
    tab = (T.p_a_given_fine if fine else T.p_a_given_x)[k]
    ok = a >= 0
    a, x = a[ok], x[ok]
    n = int(a.numel())
    if n < 10:
        return None
    P = torch.where((x >= 0)[:, None], tab[x.clamp(min=0)], T.p_aa[None, :])   # [n, 20]
    obs = torch.bincount(a, minlength=20).float(); exp = P.sum(0)
    dr = torch.multinomial(P, B, replacement=True, generator=gen)
    sim = torch.zeros(B, 20); sim.scatter_add_(1, dr.T, torch.ones(B, n))
    G = lambda c: (c * torch.log((c + 1e-9) / exp)).sum(-1)  # noqa: E731
    Z = lambda c: ((c - exp) / torch.sqrt(exp)).max(-1).values  # noqa: E731
    return {"n": n, "p": float(((G(sim) >= G(obs)).sum() + 1) / (B + 1)), "p_max": float(((Z(sim) >= Z(obs)).sum() + 1) / (B + 1)),
            "p_res": (((sim >= obs).sum(0).float() + 1) / (B + 1)).tolist(), "log2_res": torch.log2((obs + 0.5) / (exp + 0.5)).tolist()}


def score_column(T: Tables, res_tok_col: torch.Tensor, B=2000, seed=0):
    """res_tok_col [n_bins, Kr] packed tokens (keys ascending) for one column and pole."""
    gen = torch.Generator().manual_seed(seed)
    gen_cond = torch.Generator().manual_seed(1_000_003 + seed)
    out = {}
    for name, b in LEVELS.items():
        rows = dedup(T.rows(res_tok_col[b]), T.cluster)
        lvl = {"n_samples": int((res_tok_col[b] >= 0).sum()), "n_clusters": int(rows.numel())}
        if rows.numel() >= 10:
            lvl["aa"] = score_aa(T, rows, B, gen)
            for k in DESC:
                lvl[k] = score_desc(T, rows, k, B, gen, gen_cond)
            if name == "peak":
                lvl["aa_given"] = {k: score_aa_given(T, rows, k, B, gen_cond) for k in DESC}
                lvl["aa_given_fine"] = {k: score_aa_given(T, rows, k, B, gen_cond, fine=True) for k in DESC}
        out[name] = lvl
    return out


ORDINAL = {"burial3"}


def calls(scores: list[dict], alpha=0.01, rule="v1"):
    """Per-column calls from score_column outputs, with BH across columns x {structural, composition}.

    rule "v1": results/plan_ruler_v1.md as committed. rule "v1.1" (after the synthetic gate, 2026-11-05):
      - the primary descriptor's p is multiplied by the number of descriptors scored (it is a minimum of 3);
      - composition: residue identity tested against the residues expected given each sample's state on the
        primary descriptor (P(a | x); the composition-matched null in the other direction), and at least one
        residue with p < 0.01 and log2 >= 1 against that expectation (replaces effective number <= 4, which
        measures dominance, not enrichment); residue modes are reported. Residues that a state explains
        are not counted as a residue job;
      - several modes: a second state counts if it is enriched among the samples outside the top mode
        (conditional test, p < 0.01, log2 >= 0.5); for an ordinal descriptor, not a level adjacent to the top.
    v1.2 (after v1.1 on a fresh draw): as v1.1, but the residue test conditions on the finest value of the
      primary descriptor (burial: the raw C-beta count; the extreme of the count is Gly/Ala-rich, which the
      tercile did not absorb) and uses the largest standardised residue excess as its statistic.
    """
    assert rule in ("v1", "v1.1", "v1.2")
    agk = {"v1.1": ("aa_given", "p"), "v1.2": ("aa_given_fine", "p_max")}.get(rule)
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
        tests.append((d["p"] if rule == "v1" else min(1.0, d["p"] * len(cand)), i, "struct"))
        if rule == "v1" and pk["aa"]:
            tests.append((pk["aa"]["p"], i, "comp"))
        elif rule != "v1" and pk.get(agk[0], {}).get(prim):
            tests.append((pk[agk[0]][prim][agk[1]], i, "comp"))
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
        aa = pk["aa"] if rule == "v1" else pk.get(agk[0], {}).get(s["primary"])
        res_modes = [AA[r] for r in range(20) if aa and aa["p_res"][r] < 0.01 and aa["log2_res"][r] >= 1.0]
        if rule == "v1":
            specific = aa is not None and aa["eff_residues"] <= 4.0 and (i, "comp") in passed
        else:
            specific = bool(res_modes) and (i, "comp") in passed
        d = pk[s["primary"]]
        modes = [x for x in range(len(d["log2"])) if d["p_state"][x] < 0.01 and d["log2"][x] >= 0.5] if st else []
        if rule != "v1" and st and modes and d.get("cond"):
            top, cd = d["cond"]["top"], d["cond"]
            second = [x for x in range(len(cd["log2"])) if x != top and cd["p_state"][x] < 0.01 and cd["log2"][x] >= 0.5
                      and not (s["primary"] in ORDINAL and abs(x - top) == 1)]
            modes = sorted({top, *second})
        c.update({"primary": s["primary"], "modes": modes, "aa_top": pk["aa"]["top"], "eff_residues": round(pk["aa"]["eff_residues"], 2),
                  "residue_modes": res_modes})
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
