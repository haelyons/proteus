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
import resource
import queue
import threading
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.core import Atlas, AtlasConfig, gemm_paths, result_digests  # noqa: E402
from atlas.columns import attach_sources, parse_sources  # noqa: E402
from atlas.data import SwissProt, shuffled_order, plan_digest  # noqa: E402
from atlas.esm import flops_per_batch, load, pilot  # noqa: E402
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


def shard_arg(value):
    try:
        i, n = map(int, value.split("/"))
        if not 0 <= i < n:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError("shard must be i/N with 0 <= i < N and N >= 1")
    return [i, n]


def main():
    process_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8M")
    ap.add_argument("--columns", default="neurons_pre")
    ap.add_argument("--control", default="none", choices=["none", "shuffled", "random"])
    ap.add_argument("--control-seed", type=int, default=None,
                    help="seed of the weight shuffle / re-init only (default --seed); the stream, hashes and plan keep --seed")
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
    ap.add_argument("--shard", type=shard_arg, default=None)
    ap.add_argument("--pilot-out", default=None)
    ap.add_argument("--pilot-only", action="store_true")
    ap.add_argument("--pilot-from", default=None)
    a = ap.parse_args()
    if a.pilot_only and not a.pilot_out:
        ap.error("--pilot-only requires --pilot-out")
    if a.pilot_from and (a.edges_from or a.pilot_out):
        ap.error("--pilot-from is mutually exclusive with --edges-from and --pilot-out")
    if a.shard and a.shard[1] > 1 and not a.pilot_from:
        ap.error("--shard with N > 1 requires --pilot-from")
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
    model, tok, info = load(a.model, a.control, dtype=dtype, device=dev, seed=a.seed if a.control_seed is None else a.control_seed)
    info["control_seed"] = a.seed if a.control_seed is None else a.control_seed
    vocab = tok.get_vocab()
    L, D = info["n_layers"], info["d_ff"]
    try:
        sources = parse_sources(a.columns, D)
    except ValueError as e:
        ap.error(str(e))
    multi = len(sources) != 1 or sources[0].name != "neurons_pre"
    plan = ds.plan(order, a.tokens_per_batch, a.window, a.seed, shuffle_batches=not a.sorted_batches)
    if a.max_batches:
        plan = plan[: a.max_batches]
    n_total = int(sum(min(int(ds.len[i]), ds.max_res) for grp in plan for i in grp))
    plan_sha256 = plan_digest(plan)
    t_load = time.time() - t0

    cfg = AtlasConfig(K=a.K, Kr=a.Kr, backend=a.backend, seed=a.seed, reservoir=not a.no_reservoir,
                      subsample=not a.no_subsample, gram=not a.no_gram and not a.no_subsample, naive=a.naive,
                      stats_impl=a.stats_impl, merge_impl=a.merge_impl, cap_frac_early=a.cap_frac_early)
    ck = latest_ckpt(out / "ckpt") if a.resume else None
    t1 = time.time()
    artifact = None
    if a.pilot_from:
        artifact = torch.load(a.pilot_from, map_location="cpu", weights_only=False)
        if artifact["args_key"].get("columns", "neurons_pre") != a.columns:
            ap.error("--pilot-from mismatch: args_key.columns")
        expected = {"plan_sha256": plan_sha256, "n_total": n_total}
        for k, value in expected.items():
            if artifact[k] != value:
                ap.error(f"--pilot-from mismatch: {k}")
        for k, value in _pilot_args_key(a).items():
            if k not in artifact["args_key"] or artifact["args_key"][k] != value:
                ap.error(f"--pilot-from mismatch: args_key.{k}")
    if multi and a.edges_from:
        ap.error("--edges-from supports only the single neurons_pre source; use --pilot-from for column sources")
    if ck is not None:
        st = torch.load(ck, weights_only=False)
        assert st["args_key"] == _args_key(a), "resume with different run arguments"
        states = [st["sources"][s.name] for s in sources] if multi else [st["atlas"]]
        edges = [state["E"] for state in states]
        seeds_by_source = [None] * len(sources)
        start = st["next_batch"]
    else:
        seeds_by_source = [None] * len(sources)
        if artifact is not None:
            entries = [artifact["sources"][s.name] for s in sources] if multi else [artifact]
            edges = [entry["edges"] for entry in entries]
            seeds_by_source = [(entry["top_seed"], entry["res_seed_key"])
                               if entry["top_seed"] is not None or entry["res_seed_key"] is not None else None
                               for entry in entries]
        elif a.edges_from and not a.seed_thresholds:
            src = torch.load(a.edges_from, map_location="cpu", weights_only=False)
            edges = [src["edges"] if "edges" in src else src["atlas"]["E"]]
            print(f"[atlas] bin edges from {a.edges_from} {tuple(edges[0].shape)}; no pilot", flush=True)
        else:
            pplan = ds.pilot_plan(plan, a.pilot_tokens, a.tokens_per_batch, a.seed)  # random sequences, all lengths
            batches = ds.batches(vocab, order, a.tokens_per_batch, plan=pplan)
            if multi:
                values, pilot_tok = _pilot_sources(model, sources, batches, dev)
            else:
                Hp, n_pilot = pilot(model, batches, dev, 1 << 40)
                values, pilot_tok = [Hp], pilot.last_tok
                del Hp
            if a.edges_from:
                src = torch.load(a.edges_from, map_location="cpu", weights_only=False)
                edges = [src["edges"] if "edges" in src else src["atlas"]["E"]]
            else:
                edges = [torch.cat([pilot_edges([h.to(dev)], probs=tuple(a.edge_q)).cpu() for h in Hp])
                         for Hp in values]  # one layer on device at a time
            if a.seed_thresholds:
                from atlas.reference import pilot_seeds
                prot = pilot_tok // 1024  # protein of each pilot row: tails cluster within proteins
                for i, (Hp, E) in enumerate(zip(values, edges)):
                    parts = [pilot_seeds([h.to(dev)], E[li:li + 1].to(dev), a.K, a.Kr, n_total, groups=prot)
                             for li, h in enumerate(Hp)]
                    seeds_by_source[i] = (torch.cat([p_[0] for p_ in parts]), torch.cat([p_[1] for p_ in parts]))
            del values
            if a.seed_thresholds:
                del Hp
        start = a.plan_offset
    if a.pilot_out:
        entries = {}
        for i, (source, E) in enumerate(zip(sources, edges)):
            # Resumed states retain the applied thresholds in their internal representation.
            if ck is not None:
                state = states[i]
                top_seed = state["top_seed"].view(L, 2, source.width) if a.seed_thresholds else None
                rs = state["res_seed"]
                res_seed_key = torch.where(rs == torch.iinfo(torch.int64).min, 2**32, -rs).view(E.shape) if a.seed_thresholds else None
            else:
                seeds = seeds_by_source[i]
                top_seed, res_seed_key = seeds if seeds is not None else (None, None)
            entries[source.name] = {"edges": E, "top_seed": top_seed, "res_seed_key": res_seed_key}
        payload = {"sources": entries} if multi else entries[sources[0].name]
        payload.update(n_total=n_total, n_batches=len(plan), plan_sha256=plan_sha256,
                       args_key=_pilot_args_key(a), model=info)
        atomic_save(payload, Path(a.pilot_out))
        if a.pilot_only:
            return
    t_pilot = time.time() - t1
    atlases = []
    for i, (source, E, seeds) in enumerate(zip(sources, edges, seeds_by_source)):
        current = Atlas(L, source.width, E, cfg, n_annot=len(ds.vocab) if cfg.subsample else 0, device=dev)
        if ck is not None:
            current.load_state_dict(states[i])
            print(f"[atlas] resumed from {ck.name}: batch {start}/{len(plan)}, n_seen {current.n_seen}", flush=True)
        elif seeds is not None:
            current.seed(*seeds)
            print(f"[atlas] seeded thresholds for n_total={n_total}: reservoir bins seeded "
                  f"{int((current.res_seed != torch.iinfo(torch.int64).min).sum())}/{current.res_seed.numel()}", flush=True)
        atlases.append(current)
    atlas = atlases[0]
    participating = set(range(len(sources)))
    hooks_enabled = True

    def wants(i):
        return hooks_enabled and i in participating

    def dispatch(i, l, H):
        if hooks_enabled and i in participating:
            atlases[i].observe(l, H)

    handles = attach_sources(model, sources, dispatch, wants=wants)
    tie_acc = {}
    if a.count_ties:
        orig = atlas.observe

        def observe_ties(l, H, _orig=orig):
            if l in atlas._active and l not in atlas._dense_layers:
                v = atlas.ctx.rowmap >= 0
                thr = atlas.top_s[:, -1].view(L, 2, sources[0].width)[l]
                eq = (((H.float() == thr[0]) | (-H.float() == thr[1])) & v[:, None]).sum()
                tie_acc[l] = eq
            _orig(l, H)

        def dispatch_ties(i, l, H):
            if hooks_enabled and i in participating:
                if i == 0:
                    observe_ties(l, H)
                else:
                    atlases[i].observe(l, H)

        for handle in handles:
            handle.remove()
        attach_sources(model, sources, dispatch_ties, wants=wants)

    def summary(done: bool, bi: int):
        sm = {"args": vars(a), "model": info, "coverage": ds.coverage(order), "n_excluded": n_excluded, "n_batches_plan": len(plan),
                "n_total_plan": n_total, "next_batch": bi, "complete": done, "seconds": {"load": t_load, "pilot": t_pilot,
                "stream_this_process": time.time() - t2}, "n_seen": atlas.n_seen, "stats": atlas.stats,
                "gemm_paths": gemm_paths(),
                "host": {"cpus": os.cpu_count(), "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}}
        if multi:
            sm.update(columns=a.columns, sources={source.name: source_summary(i)
                                                  for i, source in enumerate(sources)})
        sm["seconds"].update(final_save=final_s, total=time.time() - process_start)
        if a.shard is not None:
            sm.update(shard=a.shard, plan_sha256=plan_sha256, batches_done=batches_done)
        if torch.device(dev).type == "cuda":
            props = torch.cuda.get_device_properties(dev)
            sm["gpu"] = {"name": props.name, "total_memory": props.total_memory,
                         "max_memory_allocated": torch.cuda.max_memory_allocated(dev),
                         "max_memory_reserved": torch.cuda.max_memory_reserved(dev)}
            if getattr(props, "uuid", None) is not None:
                sm["gpu"]["uuid"] = str(props.uuid)
        return sm

    source_results = {}
    directions_sha256 = {}
    if multi:
        for source in sources:
            if source.name.startswith("randdir"):
                R = [source.directions(l, "cpu", torch.float32) for l in range(L)]
                sha256 = hashlib.sha256(b"".join(r.numpy().tobytes() for r in R)).hexdigest()
                directions_sha256[source.name] = sha256
                atomic_save({"seed": source.seed, "M": source.width, "R": R, "sha256": sha256},
                            out / f"directions_{source.name}.pt")

    def source_summary(i):
        source, current = sources[i], atlases[i]
        entry = {"width": source.width, "stats": current.stats, "n_seen": current.n_seen}
        result = source_results.get(source.name)
        verification = result if result is not None else current.verify_seeds()
        entry.update({k: verification[k]
                      for k in ("res_seeded_bins", "res_seed_limited_bins", "top_seed_short_groups")})
        if result is not None and a.shard is None:
            entry["digests"] = result_digests(result)
        if source.name in directions_sha256:
            entry["directions_sha256"] = directions_sha256[source.name]
        return entry

    final_s = 0.0
    batches_done = st.get("batches_done", atlas.stats["batches"]) if ck is not None else 0
    logf = open(out / "batches.jsonl", "a")
    ckpt_s, batch_s, last_ckpt_s = 0.0, 0.0, None

    def save_ckpt(next_batch):
        nonlocal ckpt_s, last_ckpt_s
        t = time.time()
        state = ({"sources": {source.name: current.state_dict(include_gram=True)
                              for source, current in zip(sources, atlases)}} if multi else
                 {"atlas": atlas.state_dict(include_gram=a.shard is not None)})
        atomic_save({**state, "next_batch": next_batch,
                     "batches_done": batches_done, "args_key": _args_key(a),
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
        indices = [bi for bi in range(start, len(plan))
                   if a.shard is None or bi % a.shard[1] == a.shard[0]]

        def build_batches():
            batches = ds.batches(vocab, order, a.tokens_per_batch, plan=[plan[bi] for bi in indices])
            for bi in indices:
                build_start = time.time()
                b = next(batches)
                ctx_host = atlas.make_ctx(b.n_rows, b.valid_rows, b.tok, b.ann, device="cpu") if a.prefetch > 0 else None
                yield bi, b, ctx_host, time.time() - build_start

        src_iter = iter(_prefetch(build_batches(), a.prefetch) if a.prefetch > 0 else build_batches())
        while True:
            wait_start = time.time()
            try:
                bi, b, ctx_host, build_s = next(src_iter)
            except StopIteration:
                break
            wait_s = time.time() - wait_start
            shard_pos = bi if a.shard is None else bi // a.shard[1]
            tb = time.time()
            ids, am = b.input_ids.to(dev, non_blocking=True), b.attention_mask.to(dev, non_blocking=True)
            fwd_s = None
            if a.pair_forward or (a.pair_every and bi % a.pair_every == 0):  # same batch, hooks off: the baseline
                hooks_enabled = False
                sync(dev)
                tf0 = time.time()
                model(input_ids=ids, attention_mask=am)
                sync(dev)
                fwd_s = time.time() - tf0
                hooks_enabled = True
            prof = None
            if bi in a.profile_at:
                acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if str(dev).startswith("cuda") else [])
                prof = torch.profiler.profile(activities=acts)
                prof.__enter__()
            ta0 = time.time()
            ctx = ctx_host.to(dev) if ctx_host is not None else atlas.make_ctx(b.n_rows, b.valid_rows, b.tok, b.ann)
            t_ctx = time.time() - ta0
            if ctx_host is None:
                build_s += t_ctx
            before = dict(atlas.stats)
            n_before = atlas.n_seen
            tie_acc.clear()
            participating = set(range(len(sources)))
            for current in atlases:
                current.begin_batch(ctx)
            model(input_ids=ids, attention_mask=am)
            ties = int(sum(tie_acc.values())) if tie_acc else None
            redos = [current.end_batch() for current in atlases]
            redo = sorted({l for layers in redos for l in layers})
            if redo:
                participating = {i for i, layers in enumerate(redos) if layers}
                for i in sorted(participating):
                    atlases[i].begin_batch(ctx, redo_layers=redos[i])
                model(input_ids=ids, attention_mask=am)
                for i in sorted(participating):
                    assert atlases[i].end_batch() == []
            gram_flushed = shard_pos % a.gram_flush_every == a.gram_flush_every - 1
            if gram_flushed:
                for current in atlases:
                    current.flush_gram()
            sync(dev)
            atlas_s = time.time() - ta0
            if prof is not None:
                prof.__exit__(None, None, None)
                _write_profile(prof, out / f"profile_batch_{bi}.txt", dev)
                print(f"[atlas] profiled batch {bi} -> profile_batch_{bi}.txt", flush=True)
            s = atlas.stats
            elems = 2 * ctx.n_v * sources[0].width * L
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
            if multi:
                rec["redo_layers_by_source"] = {source.name: layers for source, layers in zip(sources, redos)}
            rec.update(shard=a.shard, shard_pos=shard_pos, build_s=build_s, wait_s=wait_s)
            if torch.device(dev).type == "cuda":
                rec.update(gpu_mem_alloc=torch.cuda.memory_allocated(dev),
                           gpu_mem_reserved=torch.cuda.memory_reserved(dev))
            if ties is not None and not redo and n_before:
                rec["ties_at_tau_frac"] = ties / elems
            n_done += 1
            batches_done += 1
            last_ckpt_s = None
            if (shard_pos + 1) % a.ckpt_every == 0:
                save_ckpt(bi + 1)
            rec["ckpt_s"] = last_ckpt_s
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            batch_s = max(batch_s, rec["seconds"]) if n_done > 2 else batch_s
            if a.deadline and bi + 1 < len(plan) and time.time() > a.deadline - 2 * (batch_s + max(ckpt_s, 30.0)):
                if (shard_pos + 1) % a.ckpt_every:
                    save_ckpt(bi + 1)
                print(f"[atlas] deadline approaching: stopped cleanly at batch {bi + 1}; rerun with --resume", flush=True)
                return
            if a.stop_after and n_done >= a.stop_after:
                print(f"[atlas] stopping after {n_done} batches (simulated cap)", flush=True)
                (out / "run.json").write_text(json.dumps(summary(False, bi + 1), indent=1))
                return
    logf.close()
    tf = time.time()
    output_paths = []
    for source, current in zip(sources, atlases):
        if a.shard is None:
            res = current.result()
            res.update({"model": info, "coverage": ds.coverage(order), "vocab": ds.vocab, "order_seed": a.seed,
                        "n_total_plan": n_total, "n_batches": len(plan)})
        else:
            res = current.verify_seeds()
            res["gram_rows"] = current.gram_rows
        stem = "shard" if a.shard is not None else "atlas"
        output_path = out / (f"{stem}_{source.name}.pt" if multi else f"{stem}.pt")
        if a.shard is None:
            atomic_save(res, output_path)
        else:
            payload = {"atlas": current.state_dict(include_gram=True), "shard": a.shard,
                       "plan_sha256": plan_sha256, "n_batches_plan": len(plan), "n_total_plan": n_total,
                       "batches_done": batches_done, "args_key": _args_key(a),
                       "model": info, "coverage": ds.coverage(order), "vocab": ds.vocab,
                       "order_seed": a.seed}
            if multi:
                payload["source"] = source.name
            atomic_save(payload, output_path)
        source_results[source.name] = res
        output_paths.append(output_path)
    res = source_results[sources[0].name]
    output_path = output_paths[0]
    final_s = time.time() - tf
    digests = result_digests(res) if a.shard is None else None
    sm = summary(True, len(plan))
    size_key = "atlas_pt_GB" if a.shard is None else "shard_pt_GB"
    if a.shard is None:
        sm["digests"] = digests
    sm.update({"final_save_s": final_s, size_key: output_path.stat().st_size / 1e9,
               "gram_rows": res.get("gram_rows"), "res_seeded_bins": res["res_seeded_bins"],
               "res_seed_limited_bins": res["res_seed_limited_bins"], "top_seed_short_groups": res["top_seed_short_groups"]})
    (out / "run.json").write_text(json.dumps(sm, indent=1))
    print(f"[atlas] done: final save {final_s:.1f}s ({sm[size_key]:.2f} GB); seed verification: "
          f"{res['res_seed_limited_bins']} of {res['res_seeded_bins']} seeded bins short, top-K groups short {res['top_seed_short_groups']}", flush=True)
    print(json.dumps({k: v for k, v in summary(True, len(plan)).items() if k in ("model", "coverage", "seconds", "n_seen")}, indent=1))
    print("stats", atlas.stats)


@torch.no_grad()
def _pilot_sources(model, sources, batches, device):
    bufs = [[[] for _ in model.encoder.layer] for _ in sources]
    rows = None

    def sink(i, l, H):
        bufs[i][l].append(H.index_select(0, rows).cpu())

    handles = attach_sources(model, sources, sink)
    toks = []
    try:
        for b in batches:
            rows = b.valid_rows.to(device)
            model(input_ids=b.input_ids.to(device), attention_mask=b.attention_mask.to(device))
            toks.append(b.tok)
    finally:
        for handle in handles:
            handle.remove()
    return [[torch.cat(layer) for layer in source] for source in bufs], torch.cat(toks)


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
    key = {k: getattr(a, k) for k in keep}
    if getattr(a, "control_seed", None) is not None:   # only when set, so earlier checkpoints still resume
        key["control_seed"] = a.control_seed
    if getattr(a, "shard", None) is not None:
        key["shard"] = a.shard
    if a.columns != "neurons_pre":
        key["columns"] = a.columns
    return key


def _pilot_args_key(a) -> dict:
    keep = ["model", "control", "control_seed", "dtype", "tokens_per_batch", "window",
            "sorted_batches", "pilot_tokens", "K", "Kr", "seed", "max_seqs", "max_batches",
            "edge_q", "seed_thresholds"]
    key = {k: getattr(a, k) for k in keep}
    if a.columns != "neurons_pre":
        key["columns"] = a.columns
    return key


if __name__ == "__main__":
    main()
