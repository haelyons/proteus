"""Compare two run_atlas checkpoints (or atlas.pt files) of the same stream: exact fields must
be bitwise equal, floating sums are reported by max relative difference. Used to show that two
implementations (e.g. stats v1 vs v2, merge loop vs stacked) give the same atlas on real data.

  python scripts/compare_atlas.py A/ckpt/ckpt_160.pt B/ckpt/ckpt_160.pt [--out cmp.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.core import result_digests  # noqa: E402

EXACT = ["top_s", "top_t", "top_k", "res_s", "res_t", "res_v", "sub_bin", "enr", "ann_tot", "E", "top_seed", "res_seed",
         "top_val", "top_tok", "res_tok", "res_val", "res_key", "edges"]
CLOSE = ["sub_sum", "sub_sq", "gram"]
SCALAR = ["n_seen", "n_sub", "gram_rows"]


def load(p):
    st = torch.load(p, map_location="cpu", weights_only=False)
    return st.get("atlas", st)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--digests", metavar="RUN_JSON")
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--out")
    args = ap.parse_args()
    if bool(args.b) == bool(args.digests):
        ap.error("provide either a second atlas or --digests RUN_JSON")
    if not 0 <= args.rtol < float("inf"):
        ap.error("--rtol must be finite and non-negative")
    a = load(args.a)
    rep = {"a": args.a, "b": args.b or args.digests, "fields": {}, "missing": [],
           "exact_all": True, "close_all": True, "scalars_all": True, "rtol": args.rtol}
    if args.digests:
        b = json.loads(Path(args.digests).read_text())
        da, db = result_digests(a), b["digests"]
        for k in sorted(da.keys() | db.keys()):
            if k not in da or k not in db:
                rep["missing"].append(k)
                rep["exact_all"] = False
            else:
                eq = da[k] == db[k]
                rep["fields"][k] = {"exact": eq, "a": da[k], "b": db[k]}
                rep["exact_all"] &= eq
        rep["close_checked"] = False
        compare_scalars(a, b, [k for k in ("n_seen", "gram_rows") if k in b], rep)
    else:
        b = load(args.b)
        compare_files(a, b, rep)
        compare_scalars(a, b, SCALAR, rep)
    rep["ok"] = bool(rep["exact_all"] and rep["close_all"] and rep["scalars_all"])
    report = json.dumps(rep, indent=1)
    print(report)
    if args.out:
        Path(args.out).write_text(report)
    sys.exit(0 if rep["ok"] else 1)


def compare_files(a, b, rep):
    for k in EXACT:
        if (k in a) != (k in b):
            rep["missing"].append(k)
            rep["exact_all"] = False
        if k in a and k in b:
            x, y = a[k], b[k]
            same_shape = x.shape == y.shape
            equal = x == y if same_shape else None
            if same_shape and x.is_floating_point() and y.is_floating_point():
                equal |= torch.isnan(x) & torch.isnan(y)
            eq = same_shape and bool(equal.all())
            rep["fields"][k] = {"exact": eq, "n_diff": None if eq or not same_shape else int((~equal).sum())}
            rep["exact_all"] &= eq
    for k in CLOSE:
        if (k in a) != (k in b):
            rep["missing"].append(k)
            rep["close_all"] = False
        if k in a and k in b:
            x, y = a[k].double(), b[k].double()
            if x.shape != y.shape:
                rel = float("inf")
            elif x.numel() == 0:
                rel = 0.0
            else:
                rel = float((x - y).abs().max() / y.abs().max().clamp(min=1e-300))
            eq = rel < rep["rtol"] or (rel == 0 and rep["rtol"] == 0)
            rep["fields"][k] = {"bitwise": bool(torch.equal(x, y)), "max_rel_diff": rel, "ok": eq}
            rep["close_all"] &= eq


def compare_scalars(a, b, keys, rep):
    for k in keys:
        if (k in a) != (k in b):
            rep["missing"].append(k)
            rep["scalars_all"] = False
        if k in a and k in b:
            rep["fields"][k] = {"a": a[k], "b": b[k], "equal": a[k] == b[k]}
            rep["scalars_all"] &= a[k] == b[k]


if __name__ == "__main__":
    main()
