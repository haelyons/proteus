"""Per-column comparison statistics, independent of the columns pooled for BH."""
from __future__ import annotations

import math

import torch

from atlas.ruler import AA, DESC


def column_stats(score: dict) -> dict:
    pk = score["peak"]
    out = {"insufficient": True, "n_clusters_peak": pk.get("n_clusters", 0),
           **dict.fromkeys(("primary", "n_desc", "p_struct", "p_struct_corr", "primary_jsd", "p_res", "residue_modes"))}
    cand = [(k, pk[k]) for k in DESC if pk.get(k)]
    if "aa" not in pk or not cand:
        return out

    def jsd_excess(d):
        q = torch.tensor(d["obs"]) / max(d["n"], 1); e = torch.tensor(d["exp"]) / max(d["n"], 1)
        m = 0.5 * (q + e)
        kl = lambda u, v: float((u[u > 0] * torch.log2(u[u > 0] / v[u > 0])).sum())  # noqa: E731
        return 0.5 * kl(q, m) + 0.5 * kl(e, m)

    prim, d = max(cand, key=lambda kd: (-math.log10(kd[1]["p"]), jsd_excess(kd[1])))
    aa = pk.get("aa_given_fine", {}).get(prim)
    modes = [AA[r] for r in range(20) if aa and aa["p_res"][r] < 0.01 and aa["log2_res"][r] >= 1.0]
    out.update({"insufficient": False, "primary": prim, "n_desc": len(cand), "p_struct": d["p"],
                "p_struct_corr": min(1.0, d["p"] * len(cand)), "primary_jsd": jsd_excess(d),
                "p_res": aa["p_max"] if aa else None, "residue_modes": modes})
    return out


def share_ci(k: int, n: int, level=0.95) -> tuple[float, float]:
    """Equal-tailed Clopper-Pearson interval, by inversion of binomial tails."""
    if not 0 <= k <= n or not 0 < level < 1:
        raise ValueError("require 0 <= k <= n and 0 < level < 1")
    if n == 0:
        return 0.0, 1.0
    tail = (1 - level) / 2

    def invert(count, upper):
        # Evaluate the shorter tail; log probabilities avoid overflow at large n.
        indices = range(count, n + 1) if upper else range(count + 1)
        target = tail
        if len(indices) > (n + 1) / 2:
            indices = range(count) if upper else range(count + 1, n + 1)
            upper = not upper
            target = 1 - tail
        log_factorial = math.lgamma(n + 1)
        coeff = [(i, log_factorial - math.lgamma(i + 1) - math.lgamma(n - i + 1)) for i in indices]
        lo, hi = 0.0, 1.0
        for _ in range(60):
            p = (lo + hi) / 2
            lp, lq = math.log(p), math.log1p(-p)
            prob = math.fsum(math.exp(c + i * lp + (n - i) * lq) for i, c in coeff)
            if (prob < target) == upper:
                lo = p
            else:
                hi = p
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else invert(k, True)
    upper = 1.0 if k == n else invert(k, False)
    return lower, upper


def summarise(records: list[dict]) -> dict:
    scored = [r for r in records if not r["insufficient"]]
    n, total = len(scored), len(records)
    k = sum(r["p_struct_corr"] <= 0.01 for r in scored)
    jsd = torch.tensor([r["primary_jsd"] for r in scored], dtype=torch.float64)
    qs = torch.quantile(jsd, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.float64)).tolist() if n else [None] * 5
    return {"n_sampled": total, "n_scored": n, "k_struct": k, "share_struct": k / n if n else None,
            "share_struct_ci": list(share_ci(k, n)), "share_struct_all": k / total if total else None,
            "jsd": {"n": n, "mean": float(jsd.mean()) if n else None,
                    **dict(zip(("q10", "q25", "q50", "q75", "q90"), qs))},
            "share_residue_mode": sum(bool(r["residue_modes"]) for r in scored) / n if n else None}
