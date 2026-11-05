"""Measurement #7 (part 2): align the ubiquitin deep mutational scan (Mavor et al. 2016, eLife 15802) with
human ubiquitin and summarise its conditions. Reads the per-condition CSVs of github.com/fraser-lab/pubs2014
("<WT><pos><MUT> , <fitness or -->"); never loads its pickles.

  python scripts/dms_align.py --dir path/to/pubs2014 --out dir
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

HUMAN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
AA = "ACDEFGHIKLMNPQRSTVWY"
SDS = {"DMSO": "DMSO_mean_stdev.csv", "Caffeine": "Caffeine_stdev.csv", "DTT": "DTT_stdev.csv", "HU": "HU_stdev.csv",
       "MG132": "MG132_stdev.csv"}
REPS = 3   # biological replicates per condition (Mavor et al. 2016)
CONDS = {"DMSO": "DMSO_floored.csv", "Caffeine": "Caffeine_floored.csv", "DTT": "DTT_floored.csv",
         "HU": "HU_floored.csv", "MG132": "MG132_floored.csv"}


def read(path):
    out = {}
    for ln in open(path):
        m = re.match(r"\s*([A-Z])(\d+)([A-Z]+)\s*,\s*(\S+)", ln)
        if not m:
            continue
        wt, pos, mut, v = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        out[(pos, wt, "*" if mut == "STOP" else mut)] = None if v == "--" else float(v)
    return out


def pearson(a, b):
    a, b = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = {c: read(Path(a.dir) / f) for c, f in CONDS.items()}
    sha = {f: hashlib.sha256((Path(a.dir) / f).read_bytes()).hexdigest() for f in CONDS.values()}
    wt = {}
    for (pos, w, _m) in D["DMSO"]:
        wt[pos] = w
    yeast = "".join(wt[p] for p in sorted(wt))
    diff = [(p, HUMAN[p - 1], yeast[p - 1]) for p in range(1, 77) if HUMAN[p - 1] != yeast[p - 1]]
    miss = [k for k in D["DMSO"] if k[2] != "*" and k[2] != k[1]]
    scored = {c: sum(1 for k in miss if D[c].get(k) is not None) for c in D}
    stops = [v for k, v in D["DMSO"].items() if k[2] == "*" and v is not None]
    human_sub = {f"{y}{p}{h}": D["DMSO"].get((p, y, h)) for p, h, y in diff}
    corr = {}
    for c in D:
        if c == "DMSO":
            continue
        ks = [k for k in miss if D["DMSO"].get(k) is not None and D[c].get(k) is not None]
        corr[c] = {"n": len(ks), "pearson": round(pearson([D["DMSO"][k] for k in ks], [D[c][k] for k in ks]), 3)}
    posmean = {}
    for c in D:
        posmean[c] = {}
        for p in range(1, 77):
            v = [D[c][k] for k in miss if k[0] == p and D[c].get(k) is not None]
            posmean[c][p] = sum(v) / len(v) if v else None
    moved = {}
    for p in range(1, 77):
        vals = [posmean[c][p] for c in D if posmean[c][p] is not None]
        if len(vals) == len(D):
            moved[p] = max(vals) - min(vals)
    # noise-aware: Cochran's Q across the five condition means of each position; SE of a position mean from
    # the replicate SDs of its alleles (SE of an allele = sd / sqrt(3)); chi-square tail with 4 df in closed form
    SD = {c: read(Path(a.dir) / f) for c, f in SDS.items()}
    # each condition mapped onto the DMSO scale by an affine fit over all missense alleles scored in both,
    # so a condition-wide shift or compression is not counted as position-specific dependence
    cal = {"DMSO": (0.0, 1.0)}
    for c in D:
        if c == "DMSO":
            continue
        ks = [k for k in miss if D["DMSO"].get(k) is not None and D[c].get(k) is not None]
        x = torch.tensor([D["DMSO"][k] for k in ks], dtype=torch.float64); y = torch.tensor([D[c][k] for k in ks], dtype=torch.float64)
        b = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()); a0 = float(y.mean() - b * x.mean())
        cal[c] = (a0, b)
    het = {}
    for p in range(1, 77):
        m, w = [], []
        for c in D:
            ks = [k for k in miss if k[0] == p and D[c].get(k) is not None and SD[c].get(k) is not None]
            if len(ks) < 5:
                break
            a0, b = cal[c]
            mu = (sum(D[c][k] for k in ks) / len(ks) - a0) / b
            se2 = sum((SD[c][k] ** 2) / REPS for k in ks) / len(ks) ** 2 / b ** 2
            m.append(mu); w.append(1.0 / max(se2, 1e-12))
        if len(m) != len(D):
            continue
        mbar = sum(wi * mi for wi, mi in zip(w, m)) / sum(w)
        Q = sum(wi * (mi - mbar) ** 2 for wi, mi in zip(w, m))
        se = [1.0 / wi ** 0.5 for wi in w]
        het[p] = {"Q": Q, "p": float(torch.exp(torch.tensor(-Q / 2)) * (1 + Q / 2)), "range": max(m) - min(m),
                  "dev": [mi - mbar for mi in m], "se": se}
    nt = len(het)
    # equivalence (two one-sided tests, 90% intervals): every condition's calibrated deviation from the pooled
    # mean inside +-margin. SE of a deviation approximated by the condition's own SE (slightly conservative).
    equiv = {str(mg): sorted(p for p, h in het.items() if all(abs(dv) + 1.645 * se <= mg for dv, se in zip(h["dev"], h["se"])))
             for mg in (0.1, 0.05)}
    sig = sorted(p for p, h in het.items() if h["p"] < 0.01 / max(nt, 1))   # Bonferroni over positions
    res = {"source": "github.com/fraser-lab/pubs2014 (Mavor et al. 2016, eLife 15802)", "sha256": sha,
           "scan_wild_type": yeast, "differences_from_human": [f"{h}{p}{y}" for p, h, y in diff],
           "rows_DMSO": len(D["DMSO"]), "missense_alleles": len(miss), "missense_scored": scored,
           "stop_median_DMSO": sorted(stops)[len(stops) // 2] if stops else None,
           "humanising_substitutions_DMSO": human_sub, "pearson_vs_DMSO": corr,
           "positions_mean_range_gt_0.1": sorted(p for p, r in moved.items() if r > 0.1),
           "calibration_to_DMSO": {c: {"offset": round(v[0], 4), "slope": round(v[1], 4)} for c, v in cal.items()},
           "positions_tested": nt, "positions_condition_dependent_bonferroni_0.01": sig,
           "positions_no_detected_condition_dependence_p_gt_0.1": sorted(p for p, h in het.items() if h["p"] > 0.1),
           "positions_equivalent_across_conditions": equiv,
           "conditions": {"primary": "DMSO", "others": [c for c in CONDS if c != "DMSO"]},
           "mean_range_by_position": {p: round(r, 3) for p, r in moved.items()}}
    (out / "dms_alignment.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("mean_range_by_position", "sha256")}))


if __name__ == "__main__":
    main()
