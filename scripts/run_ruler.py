"""Run ruler v1 over every column of an atlas result (positive pole). Writes calls keyed by column index
only; what the columns are is joined afterwards (scripts/eval_synthetic.py), so the ruler stays blind.

  python scripts/run_ruler.py --atlas synth/synthetic_atlas.pt --data swissprot.pt --desc-dir desc_subset/ \
      --uniref lists/swissprot_2026_03_uniref50.tsv.gz --out calls.json [--layer 0] [--B 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.ruler import LEVELS, Tables, calls, score_column  # noqa: E402
from scripts.synthetic_columns import load_subset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--desc-dir", required=True)
    ap.add_argument("--uniref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    data = torch.load(a.data, weights_only=False)
    T = Tables(load_subset(a.desc_dir, data, a.uniref))
    at = torch.load(a.atlas, map_location="cpu", weights_only=False)
    rt = at["res_tok"][a.layer, 0]                       # [D, B, Kr], pole 0
    scores = [score_column(T, rt[c], B=a.B, seed=c) for c in range(rt.shape[0])]
    out = calls(scores)
    for c, s in zip(out, scores):
        c["levels"] = {lv: {"n_clusters": s[lv].get("n_clusters", 0),
                            **({k: {"p": s[lv][k]["p"], "log2": [round(v, 2) for v in s[lv][k]["log2"]]}
                                for k in ("state5", "burial3", "rama4") if s[lv].get(k)})} for lv in LEVELS}
    Path(a.out).write_text(json.dumps({"calls": out, "seconds": round(time.time() - t0, 1), "residues": T.n}, indent=0))
    print(json.dumps({"columns": len(out), "seconds": round(time.time() - t0, 1),
                      "calls": {k: sum(1 for c in out if c["call"] == k) for k in sorted({c["call"] for c in out})}}))


if __name__ == "__main__":
    main()
