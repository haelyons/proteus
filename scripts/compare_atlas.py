"""Compare two run_atlas checkpoints (or atlas.pt files) of the same stream: exact fields must
be bitwise equal, floating sums are reported by max relative difference. Used to show that two
implementations (e.g. stats v1 vs v2, merge loop vs stacked) give the same atlas on real data.

  python scripts/compare_atlas.py A/ckpt/ckpt_160.pt B/ckpt/ckpt_160.pt [--out cmp.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

EXACT = ["top_s", "top_t", "top_k", "res_s", "res_t", "res_v", "sub_bin", "enr", "ann_tot", "E", "top_seed", "res_seed",
         "top_val", "top_tok", "res_tok", "res_val", "res_key", "edges"]
CLOSE = ["sub_sum", "sub_sq", "gram"]
SCALAR = ["n_seen", "n_sub", "gram_rows"]


def load(p):
    st = torch.load(p, map_location="cpu", weights_only=False)
    return st.get("atlas", st)


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    rep, ok = {"a": sys.argv[1], "b": sys.argv[2], "fields": {}}, True
    for k in EXACT:
        if k in a and k in b:
            x, y = a[k], b[k]
            if x.is_floating_point():  # NaN payloads (empty slots) compare equal to each other
                x, y = torch.nan_to_num(x.double(), nan=-7.0), torch.nan_to_num(y.double(), nan=-7.0)
            eq = x.shape == y.shape and torch.equal(x, y)
            rep["fields"][k] = {"exact": bool(eq), "n_diff": None if eq or x.shape != y.shape else int((x != y).sum())}
            ok &= eq
    for k in CLOSE:
        if k in a and k in b:
            x, y = a[k].double(), b[k].double()
            rel = float((x - y).abs().max() / y.abs().max().clamp(min=1e-300))
            rep["fields"][k] = {"bitwise": bool(torch.equal(x, y)), "max_rel_diff": rel}
            ok &= rel < 1e-5
    for k in SCALAR:
        if k in a and k in b:
            rep["fields"][k] = {"a": a[k], "b": b[k], "equal": a[k] == b[k]}
            ok &= a[k] == b[k]
    rep["ok"] = bool(ok)
    print(json.dumps(rep, indent=1))
    if "--out" in sys.argv:
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text(json.dumps(rep, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
