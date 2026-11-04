"""Redundancy of the atlas samples: for every top-K list and reservoir bin, the number of samples,
distinct proteins and distinct UniRef50 clusters. Samples from one cluster (orthologs, paralogs)
or one protein are not independent evidence, so any per-bin score has to be read next to these.

  python scripts/bin_clusters.py --atlas atlas.pt --data swissprot_2026_03.pt \
      --uniref lists/swissprot_2026_03_uniref50.tsv.gz --out bin_clusters.json [--counts-out counts.pt]
"""
from __future__ import annotations

import argparse
import gzip
import json

import torch

LEVELS = ["q50", "q90", "q99", "q99.9", "q99.99", "q99.999"]


def distinct_per_row(x: torch.Tensor) -> torch.Tensor:
    """x [..., n] int64 with -1 for empty -> number of distinct non-negative values per row."""
    s = x.sort(-1).values
    new = torch.ones_like(s, dtype=torch.bool)
    new[..., 1:] = s[..., 1:] != s[..., :-1]
    return (new & (s >= 0)).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--uniref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--counts-out", default=None)
    a = ap.parse_args()
    at = torch.load(a.atlas, map_location="cpu", weights_only=False)
    accs = torch.load(a.data, weights_only=False)["acc"]
    cl_of = {}
    with gzip.open(a.uniref, "rt") as f:
        next(f)
        for ln in f:
            acc, c = ln.rstrip("\n").split("\t")
            cl_of[acc] = c
    names = {}
    cid = torch.full((len(accs),), -1, dtype=torch.long)
    for i, acc in enumerate(accs):
        c = cl_of.get(acc) or f"self:{acc}"  # unmapped: its own cluster
        cid[i] = names.setdefault(c, len(names))
    out = {"n_clusters_in_build": len(names), "levels": LEVELS}
    counts = {}
    for name, tok in (("top", at["top_tok"]), ("res", at["res_tok"])):
        valid = tok >= 0
        seq = torch.where(valid, tok // 1024, torch.full_like(tok, -1))
        clu = torch.where(valid, cid[seq.clamp(min=0)], torch.full_like(tok, -1))
        n = valid.sum(-1)
        n_prot = distinct_per_row(seq)
        n_clu = distinct_per_row(clu)
        counts[name] = {"n": n, "proteins": n_prot, "clusters": n_clu}
        L = tok.shape[0]
        if name == "top":  # [L, 2, D]
            for p in (0, 1):
                c = n_clu[:, p].float()
                out[f"top_pole{p}"] = {
                    "median_clusters": float(c.median()), "median_proteins": float(n_prot[:, p].float().median()),
                    "frac_lt10_clusters": float((c < 10).float().mean()),
                    "by_layer_median_clusters": [float(c[l].median()) for l in range(L)],
                    "by_layer_frac_lt10": [round(float((c[l] < 10).float().mean()), 4) for l in range(L)]}
        else:  # [L, 2, D, B]
            B = tok.shape[3]
            for p in (0, 1):
                rows = {}
                for b in range(B):
                    c = n_clu[:, p, :, b].float()
                    full = n[:, p, :, b] >= tok.shape[-1]
                    rows[LEVELS[b] if b < len(LEVELS) else f"bin{b}"] = {
                        "median_samples": float(n[:, p, :, b].float().median()), "frac_full": float(full.float().mean()),
                        "median_clusters": float(c.median()), "median_proteins": float(n_prot[:, p, :, b].float().median()),
                        "frac_lt10_clusters": float((c < 10).float().mean()),
                        "by_layer_frac_lt10": [round(float((c[l] < 10).float().mean()), 4) for l in range(L)]}
                out[f"res_pole{p}"] = rows
    if a.counts_out:
        torch.save(counts, a.counts_out)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("res")}, indent=0)[:1500])
    for p in (0, 1):
        for lev, r in out[f"res_pole{p}"].items():
            print(f"res pole {p} {lev:8s}: samples {r['median_samples']:.0f} (full {r['frac_full']:.2f}), proteins {r['median_proteins']:.0f}, "
                  f"clusters {r['median_clusters']:.0f}, bins with < 10 clusters {r['frac_lt10_clusters']:.3f}")


if __name__ == "__main__":
    main()
