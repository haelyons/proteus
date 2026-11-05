"""Measurement #7 (part 1): how much of each unit's sample is ubiquitin copies.

Finds every ungapped 76-residue window in the build with >= --min-id identity to human ubiquitin (one
ubiquitin unit of UBB/UBC/RPS27A/UBA52, P0CG48 residues 1-76), anchored on exact 9-mers (8 non-overlapping
anchors: a window with <= 6 mismatches keeps at least two), then, for an atlas result, the share of each
(layer, pole, unit)'s top-K tokens and of each reservoir bin's tokens that fall inside a copy.
Tokens are packed (seq_idx, position) as in atlas.core.pack_tok.

  python scripts/ubiquitin_audit.py --data swissprot.pt [--atlas atlas.pt] --out dir
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.core import SEQ_STRIDE  # noqa: E402

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"   # human, 76 aa
YEAST_DIFF = {19: ("P", "S"), 24: ("E", "D"), 28: ("A", "S")}   # 1-based; human -> yeast (checked in the assessment)


def find_copies(seqs: list[str], min_id: int, max_res: int):
    n = len(UBQ)
    anchors = {UBQ[i:i + 9]: i for i in range(0, n - 8, 9)}
    pat = re.compile("(?=(" + "|".join(map(re.escape, anchors)) + "))")
    hits = []   # (seq_idx, start, identity)
    for si, s in enumerate(seqs):
        if not any(a in s for a in anchors):   # C-speed prefilter
            continue
        starts = set()
        for m in pat.finditer(s):
            st = m.start() - anchors[m.group(1)]
            if 0 <= st <= len(s) - n:
                starts.add(st)
        for st in sorted(starts):
            ident = sum(a == b for a, b in zip(s[st:st + n], UBQ))
            if ident >= min_id and st < max_res:
                hits.append((si, st, ident))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--atlas", default=None)
    ap.add_argument("--min-id", type=int, default=70, help="identical residues of 76 (70 = 92%%)")
    ap.add_argument("--max-res", type=int, default=1022)
    ap.add_argument("--exclude-acc", default=None, help="held-out list: copies there are reported separately")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = torch.load(a.data, weights_only=False)
    seqs, accs = d["seq"], d["acc"]
    held = set()
    if a.exclude_acc:
        held = {ln.split("\t")[0].split(",")[0].strip() for ln in open(a.exclude_acc) if ln.strip() and not ln.startswith("Entry")}
    hits = find_copies(seqs, a.min_id, a.max_res)
    exact = [h for h in hits if h[2] == len(UBQ)]
    prot = sorted({h[0] for h in hits})
    # token mask over copies (positions inside the model's window only)
    toks = set()
    for si, st, _ in hits:
        for p in range(st, min(st + len(UBQ), a.max_res)):
            toks.add(si * SEQ_STRIDE + p)
    res = {"min_identity": f"{a.min_id}/76", "copies": len(hits), "exact_copies": len(exact), "proteins": len(prot),
           "proteins_with_exact": len({h[0] for h in exact}), "held_out_proteins": sum(accs[i] in held for i in prot),
           "residues_in_copies": len(toks),
           "identity_histogram": {str(k): sum(1 for h in hits if h[2] == k) for k in range(a.min_id, len(UBQ) + 1)}}
    (out / "copies.tsv").write_text("acc\tseq_idx\tstart0\tidentity\theld_out\n" + "".join(
        f"{accs[si]}\t{si}\t{st}\t{idn}\t{int(accs[si] in held)}\n" for si, st, idn in hits))
    if a.atlas:
        at = torch.load(a.atlas, map_location="cpu", weights_only=False)
        tt = torch.tensor(sorted(toks), dtype=torch.int64)
        top = at["top_tok"]                                              # [L, 2, D, K]
        in_top = torch.isin(top, tt) & (top >= 0)
        frac_top = in_top.sum(-1).float() / (top >= 0).sum(-1).clamp(min=1)
        rt = at["res_tok"]                                               # [L, 2, D, B, Kr]
        in_res = torch.isin(rt, tt) & (rt >= 0)
        frac_res = in_res.sum(-1).float() / (rt >= 0).sum(-1).clamp(min=1)   # [L, 2, D, B]
        L = top.shape[0]
        per_layer = {}
        for l in range(L):
            f = frac_top[l].flatten()
            per_layer[str(l)] = {"units_any_top": int((f > 0).sum()), "units_top_ge_0.5": int((f >= 0.5).sum()),
                                 "units_top_all": int((f == 1).sum()), "mean_top_frac": float(f.mean()),
                                 "res_bins_ge_0.5": int((frac_res[l] >= 0.5).sum()),
                                 "mean_res_frac_by_bin": [round(float(x), 6) for x in frac_res[l].mean((0, 1))]}
        stream_frac = len(toks) / max(int(at.get("n_seen", 0)), 1)
        res.update({"atlas": a.atlas, "stream_fraction_in_copies": stream_frac, "per_layer": per_layer})
        torch.save({"frac_top": frac_top.half(), "frac_res": frac_res.half()}, out / "ubq_fractions.pt")
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_layer"}))


if __name__ == "__main__":
    main()
