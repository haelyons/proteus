"""Which proteins make early layers overflow the candidate buffer?

Runs only layers 0..L-1 of ESM-2 (early exit) on the given batches of the deterministic plan,
derives per-unit two-sided thresholds (q of +h and of -h) from reference batches, and counts, per
protein, residue x unit activations beyond them. Prints the top proteins per batch with sequence
features. CPU-friendly (two layers of 650M on ~32k tokens ~ tens of seconds).

  python scripts/diag_redo.py --data data/swissprot_2026_03.pt --batches 19 27 32 33 37 45 111 113 --ref 10 11 12 13
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.data import SwissProt, shuffled_order  # noqa: E402
from atlas.esm import load  # noqa: E402

STD = set("ACDEFGHIKLMNPQRSTVWY")


class Stop(Exception):
    pass


def features(s):
    n = len(s)
    c = collections.Counter(s)
    return {"len": n, "nonstd": sum(v for k, v in c.items() if k not in STD),
            "entropy": -sum(v / n * math.log2(v / n) for v in c.values()),
            "max_run": max((len(m.group(0)) for m in re.finditer(r"(.)\1*", s)), default=0),
            "top_res": c.most_common(1)[0][0], "top_frac": c.most_common(1)[0][1] / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="650M")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batches", type=int, nargs="+", required=True)
    ap.add_argument("--ref", type=int, nargs="+", required=True)
    ap.add_argument("--q", type=float, default=0.9999)
    ap.add_argument("--tokens-per-batch", type=int, default=32768)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ds = SwissProt(a.data)
    order = shuffled_order(len(ds), 0)
    plan = ds.plan(order, a.tokens_per_batch, 8192, 0)
    model, tok, info = load(a.model, "none", torch.float32, "cpu")
    vocab = tok.get_vocab()
    cap = {}

    def mk(li):
        def hook(_m, _i, out):
            cap[li] = out.reshape(-1, out.shape[-1]).index_select(0, cap["rows"]).to(torch.bfloat16)
            if li == a.layers - 1:
                raise Stop
        return hook
    for li in range(a.layers):
        model.encoder.layer[li].intermediate.dense.register_forward_hook(mk(li))

    def acts(b):
        bt = next(iter(ds.batches(vocab, order, a.tokens_per_batch, plan=[plan[b]])))
        cap.clear()
        cap["rows"] = bt.valid_rows
        with torch.no_grad():
            try:
                model(input_ids=bt.input_ids, attention_mask=bt.attention_mask)
            except Stop:
                pass
        return bt, [cap[li] for li in range(a.layers)]

    ref = [acts(b)[1] for b in a.ref]
    thr = []
    for li in range(a.layers):
        H = torch.cat([r[li] for r in ref])  # bf16
        k = max(1, int(round(a.q * H.shape[0])))
        tp, tn = [], []
        for c0 in range(0, H.shape[1], 256):  # column chunks keep fp32 scratch small (15 GB box)
            x = H[:, c0:c0 + 256].float()
            tp.append(torch.kthvalue(x, k, dim=0).values)
            tn.append(torch.kthvalue(-x, k, dim=0).values)
        thr.append((torch.cat(tp), torch.cat(tn)))
        del H
    n_ref = sum(r[0].shape[0] for r in ref)
    report = {"q": a.q, "ref_batches": a.ref, "ref_tokens": n_ref, "batches": {}}
    for b in a.ref[:1] + a.batches:
        bt, Hs = acts(b)
        seq_of_row = bt.tok // 1024
        rec = {"n_tokens": int(bt.valid_rows.numel()), "layers": {}}
        for li, H in enumerate(Hs):
            ex = (H > thr[li][0]) | (H < -thr[li][1])                      # [n, D] (bf16 vs fp32 promotes per element)
            per_res = ex.sum(1)                                           # units beyond per residue
            tot = int(per_res.sum())
            per_prot = collections.Counter()
            for sidx, c in zip(seq_of_row.tolist(), per_res.tolist()):
                per_prot[sidx] += c
            top = per_prot.most_common(5)
            rec["layers"][li] = {"exceed_frac": tot / ex.numel(), "expected_frac": 2 * (1 - a.q), "total": tot,
                                 "top_proteins": [{"acc": ds.d["acc"][i], "share_of_batch_exceedances": c / max(tot, 1),
                                                   "exceed_per_residue": c / min(int(ds.len[i]), ds.max_res),
                                                   **features(ds.seq[i][:ds.max_res])} for i, c in top]}
        report["batches"][b] = rec
        L0 = rec["layers"][0]
        t = L0["top_proteins"][0]
        print(f"batch {b:3d}{' (ref)' if b in a.ref else ''}: layer0 exceedance {L0['exceed_frac']:.2e} (ref-expected {L0['expected_frac']:.1e}); "
              f"top protein {t['acc']} = {t['share_of_batch_exceedances']:.0%} of exceedances, {t['exceed_per_residue']:.0f} units/residue, "
              f"len {t['len']}, nonstd {t['nonstd']}, entropy {t['entropy']:.2f}, max run {t['max_run']}, top {t['top_res']} {t['top_frac']:.0%}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
