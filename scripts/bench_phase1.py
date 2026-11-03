"""Phase 1: what does the atlas cost relative to the forward pass it rides on?

Stages, in order (each appends to <out>/phase1.jsonl the moment it finishes, and per-batch
records go to <out>/<stage>.jsonl as they happen; a cap-terminated run keeps every finished
stage and the partial batches of the running one):
  env           GPU, versions, attention implementation actually instantiated
  forward       forward-only time per ~32k-row batch vs FLOP estimate (MFU), time vs Lmax fit
  naive_topk    full per-batch topk merge every batch (baseline)
  filter_torch  monotone-threshold filter, torch compaction
  micro         isolated [32768, d_ff] bf16 on real layer activations, tau from the warmed filter
                state: naive topk vs torch filter vs Triton (first call incl. compile, tile sweep),
                ties at tau, merge cost
  filter_triton, full_torch, full_triton
Stages that would not finish before --deadline (epoch seconds) are skipped and say so.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.core import NONZERO_STATIC, Atlas, AtlasConfig, grouped_merge  # noqa: E402
from atlas.data import SwissProt, shuffled_order  # noqa: E402
from atlas.esm import attach, flops_per_batch, load, pilot  # noqa: E402
from atlas.reference import pilot_edges  # noqa: E402

PEAK = {"A10": (125, 600), "A100": (312, 1555), "H100": (989, 3350), "GH200": (989, 4000), "B200": (2250, 8000)}


class Out:
    def __init__(self, d: Path):
        self.d = d
        d.mkdir(parents=True, exist_ok=True)

    def stage(self, name, rec):
        rec = {"stage": name, "t": time.time(), **rec}
        with open(self.d / "phase1.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[atlas] {name} " + json.dumps({k: v for k, v in rec.items() if k not in ("per_batch",)})[:1500], flush=True)

    def batch(self, name, rec):
        with open(self.d / f"{name}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


def cuda_time(fn, reps=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="650M")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens-per-batch", type=int, default=32768)
    ap.add_argument("--n-batches", type=int, default=40)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--probe-layer", type=int, default=16)
    ap.add_argument("--deadline", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    O = Out(Path(a.out))
    dev = a.device
    gname = torch.cuda.get_device_name()
    peak_tf, peak_bw = next((v for k, v in PEAK.items() if k in gname), (None, None))

    ds = SwissProt(a.data)
    order = shuffled_order(len(ds), 0)
    model, tok, info = load(a.model, "none", torch.bfloat16, dev)
    attn_mod = model.encoder.layer[0].attention.self
    import transformers
    try:
        import triton
        tv = triton.__version__
    except Exception as e:  # noqa: BLE001
        tv = f"unavailable: {e}"
    O.stage("env", {"gpu": gname, "peak_bf16_tflops": peak_tf, "peak_hbm_GBps": peak_bw, "torch": torch.__version__,
                    "triton": tv, "transformers": transformers.__version__, "model": info,
                    "attn_impl_config": getattr(model.config, "_attn_implementation", None),
                    "attn_module_class": type(attn_mod).__name__,
                    "attn_module_impl_attr": getattr(attn_mod, "attn_implementation", None) or
                    getattr(getattr(attn_mod, "config", None), "_attn_implementation", None)})
    L, D = info["n_layers"], info["d_ff"]
    plan = ds.plan(order, a.tokens_per_batch, 8192, 0)
    batches = list(ds.batches(tok.get_vocab(), order, a.tokens_per_batch, plan=plan[: a.n_batches + 3]))

    def fwd(b):
        model(input_ids=b.input_ids.to(dev, non_blocking=True), attention_mask=b.attention_mask.to(dev, non_blocking=True))

    def left():
        return a.deadline - time.time() if a.deadline else 1e9

    # ---- forward only
    t_fwd = []
    with torch.no_grad():
        for b in batches[:3]:
            fwd(b)
        torch.cuda.synchronize()
        for bi, b in enumerate(batches[3:]):
            t0 = time.perf_counter()
            fwd(b)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            t_fwd.append(dt)
            d_, f_ = info["d_model"], info["d_ff"]
            lin = b.n_rows * L * 2 * (4 * d_ * d_ + 2 * d_ * f_)
            att = b.n_rows * L * 4 * b.input_ids.shape[1] * d_
            O.batch("forward", {"batch": bi, "rows": b.n_rows, "valid": int(b.valid_rows.numel()),
                                "Lmax": int(b.input_ids.shape[1]), "s": dt, "flops_linear": lin, "flops_attn": att})
    rows = [json.loads(x) for x in open(O.d / "forward.jsonl")]
    F = torch.tensor([[r["flops_linear"], r["flops_attn"], 1.0] for r in rows], dtype=torch.float64)
    y = torch.tensor([r["s"] for r in rows], dtype=torch.float64)
    coef = torch.linalg.lstsq(F, y.unsqueeze(1)).solution.squeeze(1).tolist()
    fl = sum(r["flops_linear"] + r["flops_attn"] for r in rows)
    tf = fl / sum(t_fwd) / 1e12
    mean_rows = sum(r["rows"] for r in rows) / len(rows)
    fwd_rec = {
        "batches": len(rows), "ms_per_batch_mean": 1e3 * sum(t_fwd) / len(t_fwd),
        "ms_per_32k_rows": 1e3 * sum(t_fwd) / len(t_fwd) * 32768 / mean_rows,
        "mean_rows": mean_rows, "mean_Lmax": sum(r["Lmax"] for r in rows) / len(rows), "max_Lmax": max(r["Lmax"] for r in rows),
        "valid_tokens_per_s": sum(r["valid"] for r in rows) / sum(t_fwd), "padding_frac": 1 - sum(r["valid"] for r in rows) / sum(r["rows"] for r in rows),
        "achieved_tflops_vs_estimate": tf, "mfu": tf / peak_tf if peak_tf else None,
        "pred_ms_per_batch_at_peak": 1e3 * fl / len(rows) / (peak_tf * 1e12) if peak_tf else None,
        "attn_flop_share_of_estimate": sum(r["flops_attn"] for r in rows) / fl,
        "fit_s_per_TFLOP_linear": coef[0] * 1e12, "fit_s_per_TFLOP_attn": coef[1] * 1e12, "fit_intercept_s": coef[2],
        "user_pred_s_per_32k": [1.0, 1.4], "my_pred_s_per_32k": [0.7, 0.9],
    }
    O.stage("forward", fwd_rec)

    # ---- pilot edges for reservoir modes
    torch.cuda.empty_cache()
    Hp, _ = pilot(model, iter(batches[:2]), dev, 1 << 30, store_device="cpu")
    E = torch.cat([pilot_edges([h.to(dev)]).cpu() for h in Hp])  # one layer on GPU at a time
    del Hp
    torch.cuda.empty_cache()

    modes = {
        "naive_topk": dict(reservoir=False, subsample=False, naive=True),
        "filter_torch": dict(reservoir=False, subsample=False),
        "filter_triton": dict(reservoir=False, subsample=False, backend="triton"),
        "full_torch": dict(),
        "full_triton": dict(backend="triton"),
    }
    preds = {"naive_topk": ([0.5, 1.5], [1.0, None]), "filter_torch": ([0.10, 0.20], [0.10, 0.10]),
             "filter_triton": ([0.02, 0.05], [0.02, 0.03]), "full_torch": ([0.08, 0.20], [None, None]), "full_triton": ([None, None], [None, None])}
    warm = {}

    def run_mode(name):
        est = 2.5 * sum(t_fwd) * (2.5 if name == "naive_topk" else 1.5)
        if left() < est:
            O.stage(name, {"skipped": f"budget: {left():.0f}s left < est {est:.0f}s"})
            return
        try:
            cfg = AtlasConfig(K=a.K, **modes[name])
            atlas = Atlas(L, D, E, cfg, n_annot=len(ds.vocab) if cfg.subsample else 0, device=dev)
            hs = attach(model, atlas.observe)
            ov, per = [], 0
            with torch.no_grad():
                for bi, b in enumerate(batches[3:]):
                    ctx = atlas.make_ctx(b.n_rows, b.valid_rows, b.tok, b.ann)
                    st, nb = dict(atlas.stats), atlas.n_seen
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    atlas.begin_batch(ctx)
                    fwd(b)
                    redo = atlas.end_batch()
                    if redo:
                        atlas.begin_batch(ctx, redo_layers=redo)
                        fwd(b)
                        atlas.end_batch()
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    el = 2 * ctx.n_v * D * L
                    rec = {"batch": bi, "s": dt, "overhead_vs_fwd": dt / t_fwd[bi] - 1, "redo": len(redo), "n_seen_before": nb,
                           "topk_cand_frac": (atlas.stats["top_cand"] - st["top_cand"]) / el,
                           "compacted_frac": (atlas.stats["compacted"] - st["compacted"]) / el,
                           "pred_topk_frac_iid": a.K / (nb + 1) if nb else None}
                    O.batch(name, rec)
                    ov.append(rec)
            for h in hs:
                h.remove()
            sec = ov[len(ov) // 2:]
            filt = [r for r in ov[1:] if not r["redo"] and r["pred_topk_frac_iid"]]
            O.stage(name, {"overhead_first_batch": ov[0]["overhead_vs_fwd"],
                           "overhead_second_half_mean": sum(r["overhead_vs_fwd"] for r in sec) / len(sec),
                           "redo_batches": sum(1 for r in ov if r["redo"]),
                           "topk_ratio_obs_over_pred_median": sorted(r["topk_cand_frac"] / r["pred_topk_frac_iid"] for r in filt)[len(filt) // 2] if filt and not modes[name].get("naive") else None,
                           "user_pred_overhead": preds[name][0], "my_pred_overhead": preds[name][1],
                           "nonzero_static": dict(NONZERO_STATIC)})
            if name == "filter_torch":
                warm["tau"], warm["n_seen"] = atlas.top_s.view(L, 2, D, a.K)[:, :, :, -1].clone(), atlas.n_seen
            del atlas
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            O.stage(name, {"error": f"{type(e).__name__}: {str(e)[:500]}"})
            for m in model.encoder.layer:
                m.intermediate.dense._forward_hooks.clear()
            torch.cuda.empty_cache()

    run_mode("naive_topk")
    run_mode("filter_torch")

    # ---- micro: isolated [32768, D] on real activations of the probe layer
    try:
        l = a.probe_layer
        grabbed, cur = [], {}

        def grab(li, H):
            if li == l:
                grabbed.append(H.index_select(0, cur["rows"]))
        hs = attach(model, grab)
        with torch.no_grad():
            for b in batches:
                cur["rows"] = b.valid_rows.to(dev)
                fwd(b)
                if sum(x.shape[0] for x in grabbed) >= 32768:
                    break
        for h in hs:
            h.remove()
        H = torch.cat(grabbed)[:32768].contiguous()
        del grabbed
        T = H.shape[0]
        tau = warm["tau"][l] if warm else torch.quantile(H.float()[:4096], 0.999, dim=0).expand(2, D).contiguous()
        K = a.K
        st_s = torch.full((2 * D, K), float("-inf"), device=dev)
        st_t = torch.full((2 * D, K), -1, dtype=torch.long, device=dev)
        tokv = torch.arange(T, device=dev)
        cols = torch.arange(D, device=dev)
        rowmap = torch.arange(T, device=dev)
        cap = int(0.02 * T * D)

        def naive():
            for p in (0, 1):
                v, i = torch.topk(H if p == 0 else -H, K, dim=0)
                g = (p * D + cols).expand(K, D).reshape(-1)
                grouped_merge(st_s, [st_t], g, v.float().reshape(-1), [tokv[i.reshape(-1)]], K)

        def naive_select():
            for p in (0, 1):
                torch.topk(H if p == 0 else -H, K, dim=0)

        def torch_filter():
            m = (H > tau[0]) | (H < -tau[1])
            idx = torch.nonzero_static(m, size=cap, fill_value=-1)
            return H[idx[:, 0].clamp(min=0), idx[:, 1].clamp(min=0)], m.sum()

        def torch_filter_sync():
            m = (H > tau[0]) | (H < -tau[1])
            idx = torch.nonzero(m)
            return H[idx[:, 0], idx[:, 1]]

        Hf = H.float()
        n_gt = int(((Hf > tau[0]) | (-Hf > tau[1])).sum())
        n_eq = int(((Hf == tau[0]) | (-Hf == tau[1])).sum())
        mic = {"T": T, "D": D, "layer": l, "tau_from_n_seen": warm.get("n_seen"),
               "count_gt_tau": n_gt, "count_eq_tau": n_eq,
               "pred_count_gt_tau_iid": 2 * D * T * K / (warm["n_seen"] + 1) if warm else None,
               "naive_topk_plus_merge_ms": cuda_time(naive), "naive_topk_select_only_ms": cuda_time(naive_select)}
        try:
            mic["torch_filter_nonzero_static_ms"] = cuda_time(torch_filter)
        except Exception as e:  # noqa: BLE001
            mic["torch_filter_nonzero_static_error"] = str(e)[:300]
        mic["torch_filter_nonzero_sync_ms"] = cuda_time(torch_filter_sync)
        try:
            from atlas.filter_triton import filter_candidates
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            v, r, c, n = filter_candidates(H, rowmap, tau[0], tau[1], cap)
            torch.cuda.synchronize()
            mic["triton_first_call_ms_incl_compile"] = 1e3 * (time.perf_counter() - t0)
            n = int(n)
            mic["triton_survivors"] = n
            mic["triton_matches_count_gt"] = n == n_gt
            ref = torch.nonzero((Hf > tau[0]) | (-Hf > tau[1]))
            got = torch.stack((r[:n], c[:n]), 1)
            mic["triton_survivor_set_equal"] = bool(torch.equal(torch.unique(got, dim=0), torch.unique(ref, dim=0)))
            sweep = {}
            for bt, bd in [(32, 128), (64, 128), (16, 256), (128, 64), (64, 64)]:
                try:
                    sweep[f"{bt}x{bd}"] = cuda_time(lambda: filter_candidates(H, rowmap, tau[0], tau[1], cap, BLOCK_T=bt, BLOCK_D=bd))
                except Exception as e:  # noqa: BLE001
                    sweep[f"{bt}x{bd}"] = f"error: {str(e)[:120]}"
            mic["triton_tile_sweep_ms"] = sweep
            best = min((x for x in sweep.values() if isinstance(x, float)), default=None)
            mic["triton_filter_ms_best"] = best
            # contention probe: many survivors (low threshold) -> one atomic per tile, but more stores
            low = torch.full_like(tau, 1.0)
            mic["triton_ms_at_~16pct_survivors"] = cuda_time(lambda: filter_candidates(H, rowmap, low[0], low[1], int(0.5 * T * D)))
            g = torch.where(v[:n] > tau[0][c[:n]], c[:n], D + c[:n])
            sc = torch.where(v[:n] > tau[0][c[:n]], v[:n], -v[:n])
            s2, t2 = st_s.clone(), st_t.clone()
            mic["merge_survivors_ms"] = cuda_time(lambda: grouped_merge(s2, [t2], g, sc, [r[:n]], K))
        except Exception as e:  # noqa: BLE001
            mic["triton_error"] = f"{type(e).__name__}: {str(e)[:500]}"
        mic["pred_one_read_ms_at_peak_bw"] = 1e3 * T * D * 2 / (peak_bw * 1e9) if peak_bw else None
        f32 = fwd_rec["ms_per_32k_rows"]
        mic["forward_ms_per_32k_rows"] = f32
        for k in ["naive_topk_plus_merge_ms", "torch_filter_nonzero_static_ms", "torch_filter_nonzero_sync_ms", "triton_filter_ms_best"]:
            if isinstance(mic.get(k), float):
                mic[k.replace("_ms", "") + "_x_layers_frac_of_forward"] = mic[k] * L / f32
        O.stage("micro", mic)
        del H, Hf
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        O.stage("micro", {"error": f"{type(e).__name__}: {str(e)[:500]}"})

    for name in ["filter_triton", "full_torch", "full_triton"]:
        run_mode(name)
    O.stage("done", {})


if __name__ == "__main__":
    main()
