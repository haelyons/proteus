"""Measurement #6: descriptive per-layer quantities of an atlas's extreme samples, for comparing a trained
model with shuffled-weight controls without the ruler. Predictions: results/measurement_6_predictions.md.

Per layer, positive pole, median over units: effective number of residues (exp of the Shannon entropy) of the
top-K tokens and of the peak reservoir sample; share of units whose top-K come from <= 3 proteins; share whose
top-K are effectively one residue (< 1.5). Several atlases in one call; the spread across --controls is
reported next to the trained-minus-control difference.

  python scripts/controls_summary.py --data swissprot.pt --trained a8_pre/slim.pt --controls a8_shuf1/slim.pt ... --out dir
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.core import SEQ_STRIDE  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"


def residue_codes(seqs):
    """seq_idx -> (offset into a flat residue-code array); codes 0-19, 20 for anything else."""
    lut = torch.full((256,), 20, dtype=torch.uint8)
    for i, a in enumerate(AA):
        lut[ord(a)] = i
    off = torch.zeros(len(seqs) + 1, dtype=torch.int64)
    off[1:] = torch.cumsum(torch.tensor([len(s) for s in seqs]), 0)
    flat = lut[torch.frombuffer(bytearray("".join(seqs).encode()), dtype=torch.uint8).long()]
    return flat, off


def codes_of(tok, flat, off):
    q, p = tok // SEQ_STRIDE, tok % SEQ_STRIDE
    ok = tok >= 0
    idx = off[q.clamp(min=0)] + p
    c = torch.full_like(tok, -1)
    c[ok] = flat[idx[ok]].long()
    return c


def eff_number(c):
    """c [..., n] codes (-1 = empty) -> effective number of residues per row."""
    oh = torch.zeros(*c.shape[:-1], 21)
    m = c >= 0
    oh.scatter_add_(-1, c.clamp(min=0), m.float())
    f = oh / oh.sum(-1, keepdim=True).clamp(min=1)
    h = -(f * torch.log(f.clamp(min=1e-12))).sum(-1)
    return torch.exp(h)


def n_proteins(tok):
    q = torch.where(tok >= 0, tok // SEQ_STRIDE, torch.full_like(tok, -1))
    s, _ = torch.sort(q, dim=-1)
    new = torch.ones_like(s, dtype=torch.bool); new[..., 1:] = s[..., 1:] != s[..., :-1]
    return (new & (s >= 0)).sum(-1)


def summarise(path, flat, off):
    at = torch.load(path, map_location="cpu", weights_only=False)
    top = at["top_tok"][:, 0].long()                       # [L, D, K] positive pole
    peak = at["res_tok"][:, 0, :, -1].long()               # [L, D, Kr] peak bin
    L = top.shape[0]
    out = {}
    for l in range(L):
        et = eff_number(codes_of(top[l], flat, off)); ep = eff_number(codes_of(peak[l], flat, off))
        npro = n_proteins(top[l])
        out[l] = {"eff_top": float(et.median()), "eff_peak": float(ep[(peak[l] >= 0).sum(-1) >= 10].median()),
                  "share_top_le3_proteins": float((npro <= 3).float().mean()), "share_top_one_residue": float((et < 1.5).float().mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--controls", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    seqs = torch.load(a.data, weights_only=False)["seq"]
    flat, off = residue_codes(seqs)
    tr = summarise(a.trained, flat, off)
    cs = [summarise(p, flat, off) for p in a.controls]
    res = {"trained": a.trained, "controls": a.controls, "layers": {}}
    for l in tr:
        row = {}
        for q in tr[l]:
            v = [c[l][q] for c in cs]
            mean = statistics.mean(v); sd = statistics.stdev(v) if len(v) > 1 else 0.0
            diff = tr[l][q] - mean
            row[q] = {"trained": round(tr[l][q], 4), "controls": [round(x, 4) for x in v], "control_sd": round(sd, 4),
                      "difference": round(diff, 4), "sd_over_abs_difference": round(sd / abs(diff), 3) if diff else None}
        res["layers"][str(l)] = row
        print(json.dumps({"layer": l, **{q: (r["trained"], r["controls"], r["sd_over_abs_difference"]) for q, r in row.items()}}), flush=True)
    (out / "controls_summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
