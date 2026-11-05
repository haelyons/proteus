"""Score every column of one or more atlases (positive pole) with the frozen ruler v1.2 (atlas/ruler.py @ aab0c82),
using descriptors for the whole stream. Tables are built once and shared by forked workers.

BH runs across all columns of one atlas, as the rule defines it. Output per atlas: calls.json (one record per
column: layer, unit, call, primary, modes, residue modes, per-level p and log2), and call counts.

  python scripts/score_real_units.py --desc descriptors.pt --data swissprot.pt --uniref lists/..tsv.gz \
      --atlas a8_pre/slim.pt a8_shuf1/slim.pt --names pre shuf1 --out dir --workers 28
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # forked workers deadlock on an OpenMP pool created in the parent
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import glob
import gzip
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.ruler import LEVELS, Tables, calls, score_column  # noqa: E402

G = {}


def load_proteins(desc, data, uniref, max_res=1022):
    files = [desc] if Path(desc).is_file() else sorted(glob.glob(str(Path(desc) / "part_*.pt")))
    seqs = data["seq"]
    cl = {}
    with gzip.open(uniref, "rt") as f:
        next(f)
        for ln in f:
            a, c = ln.rstrip("\n").split("\t"); cl[a] = c
    prot = []
    for fp in files:
        d = torch.load(fp, weights_only=False)
        off = d["offset"]
        for j, acc in enumerate(d["acc"]):
            s, e = int(off[j]), int(off[j + 1]); e = min(e, s + max_res)
            si = int(d["seq_idx"][j])
            prot.append({"acc": acc, "seq_idx": si, "seq": seqs[si][: e - s], "ss3": d["ss3"][s:e], "plddt": d["plddt"][s:e].float(),
                         "burial": d["burial"][s:e].float(), "phi": d["phi"][s:e].float(), "psi": d["psi"][s:e].float(),
                         "cluster": cl.get(acc) or acc})
        del d
    return prot


def work(c):
    torch.set_num_threads(1)
    return c, score_column(G["T"], G["rt"][c], B=G["B"], seed=c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc", required=True, help="full descriptor file, or a directory of part_*.pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--uniref", required=True)
    ap.add_argument("--atlas", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--max-columns", type=int, default=0, help="smoke only")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = torch.load(a.data, weights_only=False)
    prot = load_proteins(a.desc, data, a.uniref)
    del data
    G["T"] = Tables(prot); G["B"] = a.B
    del prot
    print(json.dumps({"proteins_residues": G["T"].n, "tables_s": round(time.time() - t0, 1)}), flush=True)
    summary = {}
    for path, name in zip(a.atlas, a.names):
        t1 = time.time()
        at = torch.load(path, map_location="cpu", weights_only=False)
        rt = at["res_tok"][:, 0].long()                                   # [L, D, B, Kr], positive pole
        L, D = rt.shape[0], rt.shape[1]
        G["rt"] = rt.reshape(L * D, rt.shape[2], rt.shape[3])
        n = L * D if not a.max_columns else min(a.max_columns, L * D)
        with mp.get_context("fork").Pool(a.workers) as pool:
            res = dict(pool.imap_unordered(work, range(n), chunksize=16))
        scores = [res[c] for c in range(n)]
        cl = calls(scores, rule="v1.2")
        recs = []
        for c, s in zip(cl, scores):
            c["layer"], c["unit"] = divmod(c["col"], D)
            c["levels"] = {lv: {"n_clusters": s[lv].get("n_clusters", 0),
                                **({k: {"p": s[lv][k]["p"], "log2": [round(v, 2) for v in s[lv][k]["log2"]]}
                                    for k in ("state5", "burial3", "rama4") if s[lv].get(k)})} for lv in LEVELS}
            recs.append(c)
        (out / f"calls_{name}.json").write_text(json.dumps(recs))
        cnt = Counter(c["call"] for c in recs)
        by_layer = {l: dict(Counter(c["call"] for c in recs if c["layer"] == l)) for l in range(L)}
        summary[name] = {"columns": n, "calls": dict(cnt), "by_layer": by_layer, "seconds": round(time.time() - t1, 1)}
        print(json.dumps({name: summary[name]["calls"], "s": summary[name]["seconds"]}), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
