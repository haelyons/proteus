"""Why does the observed top-K candidate fraction differ from the iid prediction K/(N+1)?

Activations are computed ONCE (ESM-2, fixed random subset of FFN units per layer, all valid
tokens of the first --n-seqs sequences of the shuffled order) and then streamed through the
same top-K tracker in several orders, so only the order differs:

  bucket_all   current loader: length-bucket the whole set, shuffle batch order
  sorted       length-bucket the whole set, ascending (drift positive control)
  window64     global shuffle, bucket within windows of 64 sequences (mixed-length batches)
  tokens       tokens permuted globally (destroys within-protein correlation), same batch size

Per batch: N_before, n, #{s > tau} (what the filter keeps), #{s == tau} (exact ties at the
threshold, which strict > excludes), prediction 2*D*n*K/(N+1). fp32 and bf16 variants.
Output: <out>/diag.jsonl (one line per batch per stream) and <out>/diag_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.core import Atlas, AtlasConfig  # noqa: E402
from atlas.data import SwissProt, shuffled_order  # noqa: E402
from atlas.esm import attach, load  # noqa: E402


def capture(model, ds, vocab, order, cols, tokens_per_batch):
    """-> A [N_tok, L*C] fp32 in canonical (plan) order, seq offsets dict seq_idx -> (start, n)."""
    plan = ds.plan(order, tokens_per_batch, 8192, 0)
    rows_out, offs, cur = [], {}, {}
    pos = 0

    def sink(l, H):
        cur.setdefault(l, H.index_select(0, cur["rows"])[:, cols[l]].float())
    hs = attach(model, sink)
    with torch.no_grad():
        for b in ds.batches(vocab, order, tokens_per_batch, plan=plan):
            cur.clear()
            cur["rows"] = b.valid_rows
            model(input_ids=b.input_ids, attention_mask=b.attention_mask)
            rows_out.append(torch.cat([cur[l] for l in range(len(cols))], 1))
            s_idx = (b.tok // 1024)
            for i in b.seq_idx.tolist():
                n = int((s_idx == i).sum())
                offs[i] = (pos, n)
                pos += n
    for h in hs:
        h.remove()
    return torch.cat(rows_out), offs


def stream_orders(ds, order, offs, tokens_per_batch, seed=0):
    """-> dict name -> list of LongTensor row-index batches into A."""
    def from_plan(plan):
        out = []
        for grp in plan:
            out.append(torch.cat([torch.arange(offs[i][0], offs[i][0] + offs[i][1]) for i in grp]))
        return out
    S = {
        "bucket_all": from_plan(ds.plan(order, tokens_per_batch, 8192, seed)),
        "sorted": from_plan(ds.plan(order, tokens_per_batch, 8192, seed, shuffle_batches=False)),
        "window64": from_plan(ds.plan(order, tokens_per_batch, 64, seed)),
    }
    n_tok = sum(n for _, n in offs.values())
    mean_b = int(sum(len(b) for b in S["window64"]) / len(S["window64"]))
    perm = torch.randperm(n_tok, generator=torch.Generator().manual_seed(seed + 1))
    S["tokens"] = list(torch.split(perm, mean_b))
    return S


def run_stream(A, batches, K, dtype):
    D = A.shape[1]
    E = torch.arange(5, dtype=torch.float32).expand(1, 2, D, 5).contiguous()  # unused (no reservoir)
    at = Atlas(1, D, E, AtlasConfig(K=K, reservoir=False, subsample=False, cap_frac=0.5))
    recs = []
    for bi, idx in enumerate(batches):
        H = A.index_select(0, idx).to(dtype)
        n = H.shape[0]
        Hf = H.float()
        tau = at.top_s[:, -1].view(2, D)
        nb = at.n_seen
        gt = int(((Hf > tau[0]) | (-Hf > tau[1])).sum()) if nb else None
        eq = int(((Hf == tau[0]) | (-Hf == tau[1])).sum()) if nb else None
        ctx = at.make_ctx(n, torch.arange(n), idx.to(torch.long))  # tok ids only need to be unique
        st = at.stats["top_cand"]
        at.begin_batch(ctx)
        at.observe(0, H)
        redo = at.end_batch()
        if redo:
            at.begin_batch(ctx, redo_layers=redo)
            at.observe(0, H)
            at.end_batch()
        if nb:
            pred = 2 * D * n * K / (nb + 1)
            assert redo or at.stats["top_cand"] - st == gt, "filter count != direct count"
            recs.append({"batch": bi, "N_before": nb, "n": n, "gt": gt, "eq": eq, "pred": pred,
                         "ratio": gt / pred, "ratio_ge": (gt + eq) / pred, "redo": bool(redo)})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="8M")
    ap.add_argument("--controls", default="none,shuffled")
    ap.add_argument("--n-seqs", type=int, default=3000)
    ap.add_argument("--cols-per-layer", type=int, default=128)
    ap.add_argument("--tokens-per-batch", type=int, default=16384)
    ap.add_argument("--K", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diag.jsonl").write_text("")
    ds = SwissProt(a.data)
    order = shuffled_order(len(ds), 0)[: a.n_seqs]
    summary = {}
    for control in a.controls.split(","):
        t0 = time.time()
        model, tok, info = load(a.model, control, dtype=torch.float32, device="cpu", seed=0)
        g = torch.Generator().manual_seed(123)
        cols = [torch.randperm(info["d_ff"], generator=g)[: a.cols_per_layer] for _ in range(info["n_layers"])]
        A, offs = capture(model, ds, tok.get_vocab(), order, cols, a.tokens_per_batch)
        del model
        print(f"[{control}] captured {tuple(A.shape)} in {time.time()-t0:.0f}s", flush=True)
        S = stream_orders(ds, order, offs, a.tokens_per_batch)
        for dname, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            for sname, batches in S.items():
                recs = run_stream(A, batches, a.K, dtype)
                with open(out / "diag.jsonl", "a") as f:
                    for r in recs:
                        f.write(json.dumps({"control": control, "dtype": dname, "order": sname, **r}) + "\n")
                rs = sorted(r["ratio"] for r in recs)
                rge = sorted(r["ratio_ge"] for r in recs)
                half = recs[len(recs) // 2:]
                tot = lambda k, R: sum(r[k] for r in R)  # noqa: E731
                summary[f"{control}/{dname}/{sname}"] = {
                    "batches": len(recs), "median_ratio": rs[len(rs) // 2], "pooled_ratio": tot("gt", recs) / tot("pred", recs),
                    "pooled_ratio_2nd_half": tot("gt", half) / tot("pred", half),
                    "median_ratio_ge": rge[len(rge) // 2], "pooled_ratio_ge": (tot("gt", recs) + tot("eq", recs)) / tot("pred", recs),
                    "ties_over_gt": tot("eq", recs) / max(1, tot("gt", recs)), "max_ratio": rs[-1], "min_ratio": rs[0]}
                print(f"  {control:8s} {dname} {sname:10s} " + " ".join(f"{k}={v:.3g}" for k, v in summary[f'{control}/{dname}/{sname}'].items()), flush=True)
        del A
    (out / "diag_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
