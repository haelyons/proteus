"""Streaming top-K by baukit's algorithm on the same stream as run_atlas.py, for a like-for-like comparison.

baukit.runningstats.TopK (davidbau/baukit, inspected 2026-11-04, lines 1153-1240) keeps a [features, 5K]
buffer; for every batch it takes the per-column top-K of the batch into the buffer and, when the buffer
is full, recompresses it with top-K. No threshold, so every element of every batch goes through a top-K.
This file re-implements that algorithm (installing the package was not permitted here); it is the
algorithm being compared, not their exact code. Ties: torch.topk order (no rule).

Columns: s = +h and s = -h for every FFN unit (pre-GELU), valid tokens only, as in the atlas. Timing:
every batch also runs a hook-free forward (overhead = hooked pass / hook-free pass - 1). Output check
against an atlas.pt from run_atlas (same plan): top-K values per column as multisets; ids compared.

  python scripts/topk_baseline.py --model 650M --data swissprot.pt --exclude-acc heldout.tsv \
      --max-batches 60 --K 64 --compare atlas.pt --out dir
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.core import pack_tok  # noqa: E402,F401
from atlas.data import SwissProt, shuffled_order  # noqa: E402
from atlas.esm import load  # noqa: E402


class BaukitTopK:
    """baukit.runningstats.TopK's algorithm (largest), on a [n, F] stream with per-row ids."""

    def __init__(self, k: int):
        self.k, self.top_data, self.top_index, self.next = k, None, None, 0

    def add(self, data: torch.Tensor, index: torch.Tensor):
        if self.top_data is None:
            F = data.shape[1]
            self.top_data = torch.zeros(F, max(10, self.k * 5), dtype=data.dtype, device=data.device)
            self.top_index = torch.zeros(F, max(10, self.k * 5), dtype=torch.long, device=data.device)
        sk = min(data.shape[0], self.k)
        if self.top_data.shape[-1] < self.next + sk:  # compression: keep the top k
            td, ti = self.topk(sorted_=False)
            self.top_data[:, : td.shape[1]] = td
            self.top_index[:, : ti.shape[1]] = ti
            self.next = td.shape[1]
        cdata = data.t().clone()  # baukit clones after transpose (strided topk is slow)
        td, ti = cdata.topk(sk, sorted=False)
        self.top_data[:, self.next: self.next + sk] = td
        self.top_index[:, self.next: self.next + sk] = index[ti]
        self.next += sk

    def topk(self, sorted_=True):
        k = min(self.k, self.next)
        td, bti = self.top_data[:, : self.next].topk(k, sorted=sorted_)
        return td, torch.gather(self.top_index[:, : self.next], 1, bti)


def sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8M")
    ap.add_argument("--data", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--tokens-per-batch", type=int, default=32768)
    ap.add_argument("--window", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-acc", default=None)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--plan-offset", type=int, default=0)
    ap.add_argument("--stop-after", type=int, default=0)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--compare", default=None, help="atlas.pt from run_atlas over the same batches")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    ds = SwissProt(a.data)
    order = shuffled_order(len(ds), a.seed)
    if a.exclude_acc:  # same rule as run_atlas.py
        ex = {ln.split("\t")[0].split(",")[0].strip() for ln in open(a.exclude_acc) if ln.strip() and not ln.startswith("Entry")}
        order = order[~torch.tensor([ds.d["acc"][i] in ex for i in order.tolist()], dtype=torch.bool)]
    plan = ds.plan(order, a.tokens_per_batch, a.window, a.seed, shuffle_batches=True)
    if a.max_batches:
        plan = plan[: a.max_batches]
    model, tok, info = load(a.model, "none", dtype=dtype, device=dev, seed=a.seed)
    L, D = info["n_layers"], info["d_ff"]
    tops = [BaukitTopK(a.K) for _ in range(L)]
    cur = {"on": False}

    def mk(l):
        def hook(_m, _i, o):
            if cur["on"]:
                Hv = o.reshape(-1, D).index_select(0, cur["rows"])
                tops[l].add(torch.cat([Hv, -Hv], dim=1), cur["tok"])
        return hook
    for l in range(L):
        model.encoder.layer[l].intermediate.dense.register_forward_hook(mk(l))
    recs = []
    n_done = 0
    with torch.no_grad():
        for bi, b in enumerate(ds.batches(tok.get_vocab(), order, a.tokens_per_batch, plan=plan, start=a.plan_offset), start=a.plan_offset):
            ids, am = b.input_ids.to(dev), b.attention_mask.to(dev)
            cur["on"] = False; sync(dev); t0 = time.time(); model(input_ids=ids, attention_mask=am); sync(dev); f = time.time() - t0
            cur.update(on=True, rows=b.valid_rows.to(dev), tok=b.tok.to(dev))
            sync(dev); t0 = time.time(); model(input_ids=ids, attention_mask=am); sync(dev); h = time.time() - t0
            recs.append({"batch": bi, "fwd_only_s": f, "hooked_s": h, "overhead_vs_fwd": h / f - 1, "n_valid": int(b.valid_rows.numel()), "Lmax": int(ids.shape[1])})
            n_done += 1
            if a.stop_after and n_done >= a.stop_after:
                break
    o = [r["overhead_vs_fwd"] for r in recs]
    res = {"method": "baukit TopK algorithm (re-implemented)", "batches": len(recs), "overhead_mean": statistics.mean(o),
           "overhead_median": statistics.median(o), "fwd_mean_s": statistics.mean(r["fwd_only_s"] for r in recs)}
    vals = torch.stack([t.topk()[0].float().cpu() for t in tops])        # [L, 2D, K] sorted descending
    ids_ = torch.stack([t.topk()[1].cpu() for t in tops])
    if a.compare:
        at = torch.load(a.compare, map_location="cpu", weights_only=False)
        ours = at["top_val"].float()                                     # [L, 2, D, K]; pole 1 stored as h
        ours_s = torch.cat([ours[:, 0], -ours[:, 1]], dim=1)             # s values, [L, 2D, K], descending
        ours_t = torch.cat([at["top_tok"][:, 0], at["top_tok"][:, 1]], dim=1)
        same_vals = torch.equal(vals, ours_s)
        id_diff = int((ids_ != ours_t).sum())
        res.update({"values_equal": bool(same_vals), "max_abs_value_diff": float((vals - ours_s).abs().max()),
                    "ids_differ": id_diff, "ids_total": int(ours_t.numel())})
    (out / "batches.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
