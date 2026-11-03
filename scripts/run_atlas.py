"""End-to-end atlas run over Swiss-Prot for one ESM-2 model (+ optional control), resumable.

Outputs in --out (put it on the persistent filesystem):
  ckpt/ckpt_<batch>.pt  atomic checkpoints every --ckpt-every batches (last 2 kept): Atlas state,
                        pilot edges, batch offset, provenance. A cap-terminated run resumes with --resume.
  batches.jsonl         one line per batch, flushed as written (observed vs predicted, timings)
  run.json              summary, rewritten at every checkpoint and at the end
  atlas.pt              final result (only when the stream completes)

Prediction for the top-K candidate fraction: for iid continuous values, a new element exceeds
the K-th largest of N earlier ones with probability K/(N+1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.core import Atlas, AtlasConfig  # noqa: E402
from atlas.data import SwissProt, shuffled_order  # noqa: E402
from atlas.esm import attach, flops_per_batch, load, pilot  # noqa: E402
from atlas.reference import pilot_edges  # noqa: E402


def sync(dev):
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize()


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def latest_ckpt(d: Path):
    c = sorted(d.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return c[-1] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8M")
    ap.add_argument("--control", default="none", choices=["none", "shuffled", "random"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--tokens-per-batch", type=int, default=32768)
    ap.add_argument("--window", type=int, default=8192, help="length-bucketing window (sequences)")
    ap.add_argument("--sorted-batches", action="store_true", help="no batch shuffle within window (drift control)")
    ap.add_argument("--pilot-tokens", type=int, default=32768)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--Kr", type=int, default=16)
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--no-reservoir", action="store_true")
    ap.add_argument("--no-subsample", action="store_true")
    ap.add_argument("--no-gram", action="store_true")
    ap.add_argument("--naive", action="store_true")
    ap.add_argument("--count-ties", action="store_true", help="diagnostic: count elements equal to tau (extra pass)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after", type=int, default=0, help="testing: stop after this many batches (simulated cap)")
    ap.add_argument("--deadline", type=float, default=0.0,
                    help="epoch seconds; checkpoint and exit cleanly when the time left drops below 2x (batch + checkpoint) time")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    dev = a.device
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]

    t0 = time.time()
    ds = SwissProt(a.data)
    order = shuffled_order(len(ds), a.seed)
    if a.max_seqs:
        order = order[: a.max_seqs]
    model, tok, info = load(a.model, a.control, dtype=dtype, device=dev, seed=a.seed)
    vocab = tok.get_vocab()
    L, D = info["n_layers"], info["d_ff"]
    plan = ds.plan(order, a.tokens_per_batch, a.window, a.seed, shuffle_batches=not a.sorted_batches)
    t_load = time.time() - t0

    cfg = AtlasConfig(K=a.K, Kr=a.Kr, backend=a.backend, seed=a.seed, reservoir=not a.no_reservoir,
                      subsample=not a.no_subsample, gram=not a.no_gram and not a.no_subsample, naive=a.naive)
    ck = latest_ckpt(out / "ckpt") if a.resume else None
    t1 = time.time()
    if ck is not None:
        st = torch.load(ck, weights_only=False)
        assert st["args_key"] == _args_key(a), "resume with different run arguments"
        E = st["atlas"]["E"]
        start = st["next_batch"]
    else:
        Hp, n_pilot = pilot(model, ds.batches(vocab, order, a.tokens_per_batch, plan=plan), dev, a.pilot_tokens)
        E = torch.cat([pilot_edges([h.to(dev)]).cpu() for h in Hp])  # one layer on device at a time
        del Hp
        start = 0
    t_pilot = time.time() - t1
    atlas = Atlas(L, D, E, cfg, n_annot=len(ds.vocab) if cfg.subsample else 0, device=dev)
    if ck is not None:
        atlas.load_state_dict(st["atlas"])
        print(f"[atlas] resumed from {ck.name}: batch {start}/{len(plan)}, n_seen {atlas.n_seen}", flush=True)
    attach(model, atlas.observe)
    tie_acc = {}
    if a.count_ties:
        orig = atlas.observe

        def observe_ties(l, H, _orig=orig):
            if l in atlas._active and l not in atlas._dense_layers:
                v = atlas.ctx.rowmap >= 0
                thr = atlas.top_s[:, -1].view(L, 2, D)[l]
                eq = (((H.float() == thr[0]) | (-H.float() == thr[1])) & v[:, None]).sum()
                tie_acc[l] = eq
            _orig(l, H)
        for m in model.encoder.layer:
            m.intermediate.dense._forward_hooks.clear()
        attach(model, observe_ties)

    def summary(done: bool, bi: int):
        return {"args": vars(a), "model": info, "coverage": ds.coverage(order), "n_batches_plan": len(plan),
                "next_batch": bi, "complete": done, "seconds": {"load": t_load, "pilot": t_pilot,
                "stream_this_process": time.time() - t2}, "n_seen": atlas.n_seen, "stats": atlas.stats}

    logf = open(out / "batches.jsonl", "a")
    ckpt_s, batch_s = 0.0, 0.0

    def save_ckpt(next_batch):
        nonlocal ckpt_s
        t = time.time()
        atomic_save({"atlas": atlas.state_dict(), "next_batch": next_batch, "args_key": _args_key(a),
                     "model": info}, out / "ckpt" / f"ckpt_{next_batch}.pt")
        for old in sorted((out / "ckpt").glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[:-2]:
            old.unlink()
        (out / "run.json").write_text(json.dumps(summary(False, next_batch), indent=1))
        ckpt_s = max(ckpt_s, time.time() - t)
        print(f"[atlas] checkpoint at batch {next_batch}/{len(plan)} ({time.time() - t:.1f}s)", flush=True)
    t2 = time.time()
    n_done = 0
    bi = start
    with torch.no_grad():
        for bi, b in enumerate(ds.batches(vocab, order, a.tokens_per_batch, plan=plan, start=start), start=start):
            tb = time.time()
            ctx = atlas.make_ctx(b.n_rows, b.valid_rows, b.tok, b.ann)
            ids, am = b.input_ids.to(dev, non_blocking=True), b.attention_mask.to(dev, non_blocking=True)
            before = dict(atlas.stats)
            n_before = atlas.n_seen
            tie_acc.clear()
            atlas.begin_batch(ctx)
            model(input_ids=ids, attention_mask=am)
            ties = int(sum(tie_acc.values())) if tie_acc else None
            redo = atlas.end_batch()
            if redo:
                atlas.begin_batch(ctx, redo_layers=redo)
                model(input_ids=ids, attention_mask=am)
                assert atlas.end_batch() == []
            if bi % 100 == 99:
                atlas.flush_gram()
            sync(dev)
            s = atlas.stats
            elems = 2 * ctx.n_v * D * L
            rec = {"batch": bi, "n_seen_before": n_before, "n_valid": ctx.n_v, "rows": b.n_rows,
                   "Lmax": int(b.input_ids.shape[1]), "seconds": time.time() - tb,
                   "flops": flops_per_batch(info, b.n_rows, b.input_ids.shape[1]), "redo_layers": redo,
                   "compacted_frac": (s["compacted"] - before["compacted"]) / elems,
                   "topk_cand_frac": (s["top_cand"] - before["top_cand"]) / elems,
                   "res_cand_frac": (s["res_cand"] - before["res_cand"]) / elems,
                   "pred_topk_frac_iid": a.K / (n_before + 1) if n_before else None}
            if ties is not None and not redo and n_before:
                rec["ties_at_tau_frac"] = ties / elems
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            n_done += 1
            batch_s = max(batch_s, rec["seconds"]) if n_done > 2 else batch_s
            if (bi + 1) % a.ckpt_every == 0:
                save_ckpt(bi + 1)
            if a.deadline and bi + 1 < len(plan) and time.time() > a.deadline - 2 * (batch_s + max(ckpt_s, 30.0)):
                if (bi + 1) % a.ckpt_every:
                    save_ckpt(bi + 1)
                print(f"[atlas] deadline approaching: stopped cleanly at batch {bi + 1}; rerun with --resume", flush=True)
                return
            if a.stop_after and n_done >= a.stop_after:
                print(f"[atlas] stopping after {n_done} batches (simulated cap)", flush=True)
                (out / "run.json").write_text(json.dumps(summary(False, bi + 1), indent=1))
                return
    logf.close()
    res = atlas.result()
    res.update({"model": info, "coverage": ds.coverage(order), "vocab": ds.vocab, "order_seed": a.seed})
    atomic_save(res, out / "atlas.pt")
    (out / "run.json").write_text(json.dumps(summary(True, len(plan)), indent=1))
    print(json.dumps({k: v for k, v in summary(True, len(plan)).items() if k in ("model", "coverage", "seconds", "n_seen")}, indent=1))
    print("stats", atlas.stats)


def _args_key(a) -> dict:
    keep = ["model", "control", "data", "dtype", "max_seqs", "tokens_per_batch", "window", "sorted_batches",
            "pilot_tokens", "K", "Kr", "no_reservoir", "no_subsample", "no_gram", "naive", "seed"]
    return {k: getattr(a, k) for k in keep}


if __name__ == "__main__":
    main()
