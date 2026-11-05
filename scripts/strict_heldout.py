"""Measurement #5: a strict held-out set. Which date-held-out Swiss-Prot entries have no close relative among the
representative sequences of UniRef50 release 2021_04 (the clustering ESM-2's training set was drawn from)?

  queries : write the held-out sequences (full length) as FASTA
  classify: per query, the best hits from an MMseqs2 search (convertalis fields below) under several
            identity / coverage criteria, and the counts that survive each

What a negative result establishes, and what it does not: the targets are cluster representatives only.
UniRef50 puts a sequence in a cluster when it has >= 50% identity to, and >= 80% overlap with, the cluster's
seed (the longest UniRef90 seed sequence); the representative shown is not always that seed, and members
can be up to 50% different from it. "No hit >= 50% identity over >= 80% of a representative" therefore means
the query would not have joined any 2021_04 cluster through its representative; it does not exclude a
>= 50%-identical member (ESM-2 sampled UniRef90 members within UniRef50 clusters). The 30% columns are a
stricter, distance-from-the-whole-cluster proxy, not a proof.

FIELDS: query,target,fident,alnlen,qcov,tcov,evalue,bits,qlen,tlen
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

FIELDS = "query,target,fident,alnlen,qcov,tcov,evalue,bits,qlen,tlen"
CRITERIA = {  # name: (min identity, coverage field, min coverage)
    "id50_tcov80": (0.50, "tcov", 0.8),   # UniRef50-like membership through the representative
    "id50_qcov80": (0.50, "qcov", 0.8),
    "id40_cov50": (0.40, "either", 0.5),
    "id30_cov50": (0.30, "either", 0.5),
    "any_hit": (0.0, "either", 0.0),      # any hit at the search E-value
}


def held_accessions(path):
    return [ln.split("\t")[0].split(",")[0].strip() for ln in open(path) if ln.strip() and not ln.startswith("Entry")]


def cmd_queries(a):
    import torch
    d = torch.load(a.data, weights_only=False)
    idx = {acc: i for i, acc in enumerate(d["acc"])}
    held = [x for x in held_accessions(a.held) if x in idx]
    if a.max_queries:
        held = held[: a.max_queries]
    with open(a.out, "w") as f:
        for acc in held:
            f.write(f">{acc}\n{d['seq'][idx[acc]]}\n")
    print(json.dumps({"queries": len(held), "listed": len(held_accessions(a.held))}))


def cmd_classify(a):
    queries = [ln[1:].split()[0] for ln in open(a.queries) if ln.startswith(">")]
    best = {name: {} for name in CRITERIA}   # name -> query -> (fident, target)
    nh = collections.Counter()
    for ln in open(a.hits):
        q, t, fid, alnlen, qcov, tcov, ev, bits, qlen, tlen = ln.rstrip("\n").split("\t")
        fid, qcov, tcov = float(fid), float(qcov), float(tcov)
        nh[q] += 1
        for name, (mid, cf, mc) in CRITERIA.items():
            cov = max(qcov, tcov) if cf == "either" else (qcov if cf == "qcov" else tcov)
            if fid >= mid and cov >= mc and fid > best[name].get(q, (-1, ""))[0]:
                best[name][q] = (fid, t)
    desc = None
    if a.desc_accs:
        desc = {ln.strip() for ln in open(a.desc_accs) if ln.strip()}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "strict_heldout.tsv", "w") as f:
        f.write("acc\tn_hits\t" + "\t".join(f"{n}_best_id\t{n}_target" for n in CRITERIA) + ("\thas_descriptors" if desc is not None else "") + "\n")
        for q in queries:
            row = [q, str(nh[q])]
            for n in CRITERIA:
                fid, t = best[n].get(q, (None, ""))
                row += ["" if fid is None else f"{fid:.3f}", t]
            if desc is not None:
                row.append(str(int(q in desc)))
            f.write("\t".join(row) + "\n")
    surv = {n: [q for q in queries if q not in best[n]] for n in CRITERIA}
    res = {"queries": len(queries), "queries_with_any_hit": sum(1 for q in queries if nh[q]),
           "survivors (no hit meeting the criterion)": {n: len(v) for n, v in surv.items()},
           "criteria": {n: {"min_identity": c[0], "coverage": c[1], "min_coverage": c[2]} for n, c in CRITERIA.items()}}
    if desc is not None:
        res["survivors_with_descriptors"] = {n: sum(1 for q in v if q in desc) for n, v in surv.items()}
        res["queries_with_descriptors"] = sum(1 for q in queries if q in desc)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("queries"); q.add_argument("--data", required=True); q.add_argument("--held", required=True)
    q.add_argument("--out", required=True); q.add_argument("--max-queries", type=int, default=0)
    c = sp.add_parser("classify"); c.add_argument("--hits", required=True); c.add_argument("--queries", required=True)
    c.add_argument("--out", required=True); c.add_argument("--desc-accs", default=None)
    a = ap.parse_args()
    {"queries": cmd_queries, "classify": cmd_classify}[a.cmd](a)


if __name__ == "__main__":
    main()
