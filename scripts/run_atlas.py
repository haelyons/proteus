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
import hashlib
import json
import queue
import threading
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.core import Atlas, AtlasConfig, gemm_paths  # noqa: E402
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
    ap.add_argument("--cap-frac-early", type=float, default=0.0,
                    help="candidate buffer fraction for layers 0-1 (default: same as the others, 0.005)")
    ap.add_argument("--stats-impl", default="v3", choices=["v3", "v2", "v1"], help="subsample statistics: v3 Triton kernels (CUDA), v1 reference")
    ap.add_argument("--merge-impl", default="stacked", choices=["stacked", "loop"], help="reservoir bulk rows (loop = reference)")
    ap.add_argument("--no-reservoir", action="store_true")
    ap.add_argument("--no-subsample", action="store_true")
    ap.add_argument("--no-gram", action="store_true")
    ap.add_argument("--naive", action="store_true")
    ap.add_argument("--count-ties", action="store_true", help="diagnostic: count elements equal to tau (extra pass)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--gram-flush-every", type=int, default=500,
                    help="batches between fp32 device Gram -> fp64 host flushes (each moves L*D^2*4 bytes)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after", type=int, default=0, help="testing: stop after this many batches (simulated cap)")
    ap.add_argument("--seed-thresholds", action="store_true",
                    help="exact-preserving start thresholds from the pilot (top-K: 2K-th pilot value; reservoir: key ceilings)")
    ap.add_argument("--pair-forward", action="store_true", help="time a hook-free forward of each batch first (overhead per batch)")
    ap.add_argument("--pair-every", type=int, default=0, help="as --pair-forward, but only on every N-th batch (overhead on a sample)")
    ap.add_argument("--plan-offset", type=int, default=0,
                    help="start streaming at this batch of the plan with fresh state (timing comparisons of stateless paths)")
    ap.add_argument("--prefetch", type=int, default=2,
                    help="batches (and their host-side contexts) built ahead on a thread; 0 builds them inline")
    ap.add_argument("--only-acc", default=None, help="file of accessions: stream only these (e.g. the held-out set)")
    ap.add_argument("--edges-from", default=None,
                    help="atlas.pt or checkpoint whose bin edges to reuse (no pilot edges; no seeds unless --seed-thresholds)")
    ap.add_argument("--exclude-acc", default=None,
                    help="file of UniProt accessions (first column; header line 'Entry' skipped) left out of the stream, e.g. a held-out set")
    ap.add_argument("--edge-q", type=float, nargs="+", default=[0.5, 0.9, 0.99, 0.999, 0.9999],
                    help="pilot quantiles of s used as bin lower edges")
    ap.add_argument("--profile-at", type=int, nargs="*", default=[], help="batch indices to profile (torch.profiler)")
    ap.add_argument("--max-batches", type=int, default=0, help="stream only the first M batches of the plan (complete run)")
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
    n_excluded = 0
    def acc_set(path):
        return {ln.split("\t")[0].split(",")[0].strip() for ln in open(path) if ln.strip() and not ln.startswith("Entry")}
    if a.only_acc:
        inc = acc_set(a.only_acc)
        order = order[torch.tensor([ds.d["acc"][i] in inc for i in order.tolist()], dtype=torch.bool)]
        print(f"[atlas] streaming only {len(order)} of {len(inc)} listed accessions ({a.only_acc})", flush=True)
    if a.exclude_acc:
        ex = acc_set(a.exclude_acc)
        drop = torch.tensor([acc in ex for acc in ds.d["acc"]], dtype=torch.bool)
        keep = ~drop[order]
        n_excluded = int((~keep).sum())
        order = order[keep]
        print(f"[atlas] excluded {n_excluded} of {len(ex)} listed accessions ({a.exclude_acc})", flush=True)
    if a.max_seqs:
        order = order[: a.max_seqs]
    model, tok, info = load(a.model, a.control, dtype=dtype, device=dev, seed=a.seed)
    vocab = tok.get_vocab()
    L, D = info["n_layers"], info["d_ff"]
    plan = ds.plan(order, a.tokens_per_batch, a.window, a.seed, shuffle_batches=not a.sorted_batches)
    if a.max_batches:
        plan = plan[: a.max_batches]
    n_total = int(sum(min(int(ds.len[i]), ds.max_res) for grp in plan for i in grp))
    t_load = time.time() - t0

    cfg = AtlasConfig(K=a.K, Kr=a.Kr, backend=a.backend, seed=a.seed, reservoir=not a.no_reservoir,
                      subsample=not a.no_subsample, gram=not a.no_gram and not a.no_subsample, naive=a.naive,
                      stats_impl=a.stats_impl, merge_impl=a.merge_impl, cap_frac_early=a.cap_frac_early)
    ck = latest_ckpt(out / "ckpt") if a.resume else None
    t1 = time.time()
    if ck is not None:
        st = torch.load(ck, weights_only=False)
        assert st["args_key"] == _args_key(a), "resume with different run arguments"
        E = st["atlas"]["E"]
        start = st["next_batch"]
    else:
        seeds = None
        if a.edges_from and not a.seed_thresholds:
            src = torch.load(a.edges_from, map_location="cpu", weights_only=False)
            E = src["edges"] if "edges" in src else src["atlas"]["E"]
            Hp = None
            print(f"[atlas] bin edges from {a.edges_from} {tuple(E.shape)}; no pilot", flush=True)
        else:
            pplan = ds.pilot_plan(plan, a.pilot_tokens, a.tokens_per_batch, a.seed)  # random sequences, all lengths
            Hp, n_pilot = pilot(model, ds.batches(vocab, order, a.tokens_per_batch, plan=pplan), dev, 1 << 40)
            if a.edges_from:
                src = torch.load(a.edges_from, map_location="cpu", weights_only=False)
                E = src["edges"] if "edges" in src else src["atlas"]["E"]
            else:
                E = torch.cat([pilot_edges([h.to(dev)], probs=tuple(a.edge_q)).cpu() for h in Hp])  # one layer on device at a time
        if a.seed_thresholds:
            from atlas.reference import pilot_seeds
            prot = pilot.last_tok // 1024  # protein of each pilot row: tails cluster within proteins
            parts = [pilot_seeds([h.to(dev)], E[li:li + 1].to(dev), a.K, a.Kr, n_total, groups=prot)
                     for li, h in enumerate(Hp)]
            seeds = (torch.cat([p_[0] for p_ in parts]), torch.cat([p_[1] for p_ in parts]))
        del Hp
        start = a.plan_offset
    t_pilot = time.time() - t1
    atlas = Atlas(L, D, E, cfg, n_annot=len(ds.vocab) if cfg.subsample else 0, device=dev)
    if ck is not None:
        atlas.load_state_dict(st["atlas"])
        print(f"[atlas] resumed from {ck.name}: batch {start}/{len(plan)}, n_seen {atlas.n_seen}", flush=True)
    elif seeds is not None:
        atlas.seed(*seeds)
        print(f"[atlas] seeded thresholds for n_total={n_total}: reservoir bins seeded "
              f"{int((atlas.res_seed != torch.iinfo(torch.int64).min).sum())}/{atlas.res_seed.numel()}", flush=True)
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
        return {"args": vars(a), "model": info, "coverage": ds.coverage(order), "n_excluded": n_excluded, "n_batches_plan": len(plan),
                "next_batch": bi, "complete": done, "seconds": {"load": t_load, "pilot": t_pilot,
                "stream_this_process": time.time() - t2}, "n_seen": atlas.n_seen, "stats": atlas.stats,
                "gemm_paths": gemm_paths()}

    logf = open(out / "batches.jsonl", "a")
    ckpt_s, batch_s, last_ckpt_s = 0.0, 0.0, None

    def save_ckpt(next_batch):
        nonlocal ckpt_s, last_ckpt_s
        t = time.time()
        atomic_save({"atlas": atlas.state_dict(), "next_batch": next_batch, "args_key": _args_key(a),
                     "model": info}, out / "ckpt" / f"ckpt_{next_batch}.pt")
        for old in sorted((out / "ckpt").glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[:-2]:
            old.unlink()
        (out / "run.json").write_text(json.dumps(summary(False, next_batch), indent=1))
        last_ckpt_s = time.time() - t
        ckpt_s = max(ckpt_s, last_ckpt_s)
        sz = (out / "ckpt" / f"ckpt_{next_batch}.pt").stat().st_size
        print(f"[atlas] checkpoint at batch {next_batch}/{len(plan)} ({last_ckpt_s:.1f}s, {sz / 1e9:.2f} GB)", flush=True)
    t2 = time.time()
    n_done = 0
    bi = start
    with torch.no_grad():
        src_iter = ds.batches(vocab, order, a.tokens_per_batch, plan=plan, start=start)
        if a.prefetch > 0:  # batch building and host-side contexts off the critical path
            src_iter = _prefetch(((b, atlas.make_ctx(b.n_rows, b.valid_rows, b.tok, b.ann, device="cpu")) for b in src_iter), a.prefetch)
        else:
            src_iter = ((b, None) for b in src_iter)
        for bi, (b, ctx_host) in enumerate(src_iter, start=start):
            tb = time.time()
            ids, am = b.input_ids.to(dev, non_blocking=True), b.attention_mask.to(dev, non_blocking=True)
            fwd_s = None
            if a.pair_forward or (a.pair_every and bi % a.pair_every == 0):  # same batch, hooks off: the baseline
                atlas.enabled = False
                sync(dev)
                tf0 = time.time()
                model(input_ids=ids, attention_mask=am)
                sync(dev)
                fwd_s = time.time() - tf0
                atlas.enabled = True
            prof = None
            if bi in a.profile_at:
                acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if str(dev).startswith("cuda") else [])
                prof = torch.profiler.profile(activities=acts)
                prof.__enter__()
            ta0 = time.time()
            ctx = ctx_host.to(dev) if ctx_host is not None else atlas.make_ctx(b.n_rows, b.valid_rows, b.tok, b.ann)
            t_ctx = time.time() - ta0
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
            gram_flushed = bi % a.gram_flush_every == a.gram_flush_every - 1
            if gram_flushed:
                atlas.flush_gram()
            sync(dev)
            atlas_s = time.time() - ta0
            if prof is not None:
                prof.__exit__(None, None, None)
                _write_profile(prof, out / f"profile_batch_{bi}.txt", dev)
                print(f"[atlas] profiled batch {bi} -> profile_batch_{bi}.txt", flush=True)
            s = atlas.stats
            elems = 2 * ctx.n_v * D * L
            rec = {"batch": bi, "n_seen_before": n_before, "n_valid": ctx.n_v, "rows": b.n_rows,
                   "Lmax": int(b.input_ids.shape[1]), "seconds": time.time() - tb, "t_start": tb, "t_end": time.time(),
                   "fwd_only_s": fwd_s, "atlas_pass_s": atlas_s, "make_ctx_s": t_ctx,
                   "overhead_vs_fwd": (atlas_s / fwd_s - 1) if fwd_s else None, "profiled": prof is not None,
                   "gram_flushed": gram_flushed,
                   "flops": flops_per_batch(info, b.n_rows, b.input_ids.shape[1]), "redo_layers": redo,
                   "compacted_frac": (s["compacted"] - before["compacted"]) / elems,
                   "topk_cand_frac": (s["top_cand"] - before["top_cand"]) / elems,
                   "res_cand_frac": (s["res_cand"] - before["res_cand"]) / elems,
                   "bulk_rows_in": s.get("bulk_rows_in", 0) - before.get("bulk_rows_in", 0),
                   "bulk_rows_kept": s.get("bulk_rows_kept", 0) - before.get("bulk_rows_kept", 0),
                   "pred_topk_frac_iid": a.K / (n_before + 1) if n_before else None}
            if ties is not None and not redo and n_before:
                rec["ties_at_tau_frac"] = ties / elems
            n_done += 1
            last_ckpt_s = None
            if (bi + 1) % a.ckpt_every == 0:
                save_ckpt(bi + 1)
            rec["ckpt_s"] = last_ckpt_s
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            batch_s = max(batch_s, rec["seconds"]) if n_done > 2 else batch_s
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
    res.update({"model": info, "coverage": ds.coverage(order), "vocab": ds.vocab, "order_seed": a.seed,
                "n_total_plan": n_total, "n_batches": len(plan)})
    tf = time.time()
    atomic_save(res, out / "atlas.pt")
    final_s = time.time() - tf
    digests = {k: hashlib.sha256(res[k].contiguous().numpy().tobytes()).hexdigest()
               for k in ("top_val", "top_tok", "res_tok", "res_val", "res_key", "sub_bin", "enr", "ann_tot", "edges") if k in res}
    sm = summary(True, len(plan))
    sm.update({"digests": digests, "final_save_s": final_s, "atlas_pt_GB": (out / "atlas.pt").stat().st_size / 1e9,
               "gram_rows": res.get("gram_rows"), "res_seeded_bins": res["res_seeded_bins"],
               "res_seed_limited_bins": res["res_seed_limited_bins"], "top_seed_short_groups": res["top_seed_short_groups"]})
    (out / "run.json").write_text(json.dumps(sm, indent=1))
    print(f"[atlas] done: final save {final_s:.1f}s ({sm['atlas_pt_GB']:.2f} GB); seed verification: "
          f"{res['res_seed_limited_bins']} of {res['res_seeded_bins']} seeded bins short, top-K groups short {res['top_seed_short_groups']}", flush=True)
    print(json.dumps({k: v for k, v in summary(True, len(plan)).items() if k in ("model", "coverage", "seconds", "n_seen")}, indent=1))
    print("stats", atlas.stats)


def _prefetch(it, n):
    """Run the iterator on a daemon thread, n items ahead; errors are re-raised in the consumer."""
    q, end = queue.Queue(maxsize=n), object()

    def work():
        try:
            for x in it:
                q.put(x)
        except BaseException as e:  # noqa: BLE001
            q.put(e)
        q.put(end)
    threading.Thread(target=work, daemon=True).start()
    while True:
        x = q.get()
        if x is end:
            return
        if isinstance(x, BaseException):
            raise x
        yield x


def _write_profile(prof, path, dev):
    """Kernel table (key_averages) plus a per-range attribution from the Chrome trace: span vs
    busy device time per atlas.* range and the top kernels inside each (atlas/proftrace.py)."""
    try:
        import gzip
        import shutil
        from atlas.proftrace import attribute, table
        tp = path.with_suffix(".trace.json")
        prof.export_chrome_trace(str(tp))
        res = attribute(tp)
        path.with_name(path.stem + "_ranges.txt").write_text(table(res, top=10))
        path.with_name(path.stem + "_ranges.json").write_text(json.dumps(res))
        with open(tp, "rb") as fi, gzip.open(str(tp) + ".gz", "wb") as fo:
            shutil.copyfileobj(fi, fo)
        tp.unlink()
    except Exception as e:  # noqa: BLE001  (attribution is a diagnostic; never fail the run on it)
        print(f"[atlas] trace attribution failed: {e!r}", flush=True)
    ka = prof.key_averages()
    for key in (("self_cuda_time_total", "cuda_time_total") if str(dev).startswith("cuda") else ("self_cpu_time_total",)):
        try:
            path.write_text(ka.table(sort_by=key, row_limit=60))
            return
        except Exception:  # noqa: BLE001  (key names move between torch versions)
            continue
    path.write_text(ka.table(sort_by="self_device_time_total", row_limit=60))


def _args_key(a) -> dict:
    keep = ["model", "control", "data", "dtype", "max_seqs", "tokens_per_batch", "window", "sorted_batches",
            "pilot_tokens", "K", "Kr", "no_reservoir", "no_subsample", "no_gram", "naive", "seed",
            "seed_thresholds", "max_batches", "exclude_acc", "edge_q", "only_acc", "edges_from"]
    return {k: getattr(a, k) for k in keep}


if __name__ == "__main__":
    main()
