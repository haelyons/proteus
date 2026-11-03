"""Where does the ESM-2 forward pass spend its time, and do the GEMMs run in bf16?

Writes <out>/profile_forward.json (+ kernel_table.txt):
  env        parameter/buffer dtypes, attention implementation
  profile    torch.profiler over a few batches: device time by kernel class (GEMM by declared dtype,
             attention, layer norm, elementwise, other) and the top kernels by name
  forward    forward-only per-batch times with wall-clock bounds (to align with nvidia-smi samples)
  gemm       standalone GEMMs at the model's shapes: bf16 vs TF32 TFLOP/s, plus a sustained window
  tiles      Triton compaction tile sweep (early-exit kernel) on real layer activations
Every timed window carries [t_start, t_end] epoch seconds.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.data import SwissProt, shuffled_order  # noqa: E402
from atlas.esm import attach, flops_per_batch, load  # noqa: E402


def sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def classify(name: str) -> str:
    n = name.lower()
    if re.search(r"fmha|flash|attention|mem_eff|efficient", n):
        return "attention"
    if re.search(r"gemm|cutlass|xmma|s16816|h16816|matmul|cublas|sm80_|sm86_|ampere_", n):
        if "bf16" in n:
            return "gemm_bf16"
        if "tf32" in n:
            return "gemm_tf32"
        if re.search(r"f16|h16816|fp16|half", n):
            return "gemm_fp16"
        if re.search(r"sgemm|f32|fp32", n):
            return "gemm_fp32"
        return "gemm_unknown_dtype"
    if re.search(r"layer_?norm", n):
        return "layernorm"
    if re.search(r"softmax", n):
        return "softmax"
    if re.search(r"elementwise|vectorized|gelu|add|mul|copy|fill|cat|index|reduce|where|rotary|cos|sin", n):
        return "elementwise"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="650M")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-batches", type=int, default=12)
    ap.add_argument("--tokens-per-batch", type=int, default=32768)
    ap.add_argument("--sustain-s", type=float, default=15.0)
    ap.add_argument("--probe-layer", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    cuda = str(dev).startswith("cuda")
    R: dict = {}

    def save():
        (out / "profile_forward.json").write_text(json.dumps(R, indent=1))

    ds = SwissProt(a.data)
    order = shuffled_order(len(ds), 0)
    model, tok, info = load(a.model, "none", torch.bfloat16 if cuda else torch.float32, dev)
    pd = collections.Counter(str(p.dtype) for p in model.parameters())
    bd = collections.Counter(f"{n.split('.')[-1]}:{b.dtype}" for n, b in model.named_buffers())
    attn = model.encoder.layer[0].attention.self
    R["env"] = {"gpu": torch.cuda.get_device_name() if cuda else "cpu", "torch": torch.__version__, "model": info,
                "param_dtypes": dict(pd), "buffer_dtypes": dict(bd),
                "attn_impl": getattr(model.config, "_attn_implementation", None), "attn_class": type(attn).__name__,
                "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "autocast": torch.is_autocast_enabled()}
    save()
    plan = ds.plan(order, a.tokens_per_batch, 8192, 0)[: a.n_batches]
    batches = list(ds.batches(tok.get_vocab(), order, a.tokens_per_batch, plan=plan))

    def fwd(b):
        model(input_ids=b.input_ids.to(dev, non_blocking=True), attention_mask=b.attention_mask.to(dev, non_blocking=True))

    with torch.no_grad():
        for b in batches[:2]:
            fwd(b)
        sync(dev)
        # ---- profiler
        acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if cuda else [])
        pb = batches[2:5]
        t0 = time.time()
        with torch.profiler.profile(activities=acts) as prof:
            for b in pb:
                fwd(b)
            sync(dev)
        t1 = time.time()
        ka = prof.key_averages()
        rows = []
        for e in ka:
            if cuda:
                # kernel rows only: aten:: ops report the device time of the kernels they launch,
                # so summing both double-counts (phase-1 profile: 2.7 s "device" per 1.34 s batch)
                if str(getattr(e, "device_type", "")).split(".")[-1] not in ("CUDA",) and (e.key.startswith("aten::") or e.key in ("Command Buffer Full",)):
                    continue
                dt = getattr(e, "self_device_time_total", None)
                if dt is None:
                    dt = getattr(e, "self_cuda_time_total", 0)
            else:
                dt = e.self_cpu_time_total
            if dt:
                rows.append((e.key, dt / 1e3 / len(pb), e.count / len(pb)))  # ms per batch
        rows.sort(key=lambda r: -r[1])
        by = collections.Counter()
        for k, ms, _ in rows:
            by[classify(k)] += ms
        tot = sum(by.values())
        try:
            (out / "kernel_table.txt").write_text(ka.table(sort_by="self_cuda_time_total" if cuda else "self_cpu_time_total", row_limit=60))
        except Exception:  # noqa: BLE001
            (out / "kernel_table.txt").write_text("\n".join(f"{ms:10.3f} ms  {k}" for k, ms, _ in rows[:60]))
        fl = sum(flops_per_batch(info, b.n_rows, b.input_ids.shape[1]) for b in pb) / len(pb)
        lin = sum(b.n_rows for b in pb) / len(pb) * info["n_layers"] * 2 * (4 * info["d_model"] ** 2 + 2 * info["d_model"] * info["d_ff"])
        gemm_ms = sum(v for k, v in by.items() if k.startswith("gemm"))
        R["profile"] = {"t_start": t0, "t_end": t1, "batches": len(pb), "wall_ms_per_batch": 1e3 * (t1 - t0) / len(pb),
                        "device_ms_per_batch_by_class": dict(by), "device_ms_per_batch_total": tot,
                        "class_share": {k: v / tot for k, v in by.items()} if tot else {},
                        "linear_flops_per_batch": lin, "est_flops_per_batch": fl,
                        "gemm_tflops_on_gemm_time": lin / (gemm_ms / 1e3) / 1e12 if gemm_ms else None,
                        "top_kernels": [{"name": k[:200], "ms_per_batch": ms, "calls_per_batch": c, "class": classify(k)} for k, ms, c in rows[:30]]}
        save()
        print("[atlas] profile", json.dumps({k: v for k, v in R["profile"].items() if k != "top_kernels"})[:1500], flush=True)
        # ---- forward-only timing with wall-clock bounds
        recs = []
        for b in batches[5:]:
            sync(dev)
            s0 = time.time()
            fwd(b)
            sync(dev)
            s1 = time.time()
            recs.append({"t_start": s0, "t_end": s1, "s": s1 - s0, "rows": b.n_rows, "Lmax": int(b.input_ids.shape[1]),
                         "flops": flops_per_batch(info, b.n_rows, b.input_ids.shape[1])})
        R["forward"] = {"batches": recs, "tflops": sum(r["flops"] for r in recs) / sum(r["s"] for r in recs) / 1e12 if recs else None}
        save()
        print("[atlas] forward", f"{R['forward']['tflops']:.1f} TFLOP/s over {len(recs)} batches", flush=True)

        # ---- forward reproducibility: same batch twice, bitwise, for a bucketed and a padded batch
        try:
            from atlas.data import SwissProt as _SP  # noqa: F401
            probe = {}
            cap_ = {}

            def grab2(li, H):
                if li == a.probe_layer % info["n_layers"]:
                    cap_["H"] = H.index_select(0, cap_["rows"]).float().clone()
            hs = attach(model, grab2)
            mixed = ds.pilot_plan(plan, a.tokens_per_batch // 2, a.tokens_per_batch, 0)[:1]
            for nm, bb in (("bucketed", batches[5]), ("padded_mixed_lengths", next(iter(ds.batches(tok.get_vocab(), order, a.tokens_per_batch, plan=mixed))))):
                outs = []
                for _ in range(3):
                    cap_["rows"] = bb.valid_rows.to(dev)
                    fwd(bb)
                    sync(dev)
                    outs.append(cap_["H"])
                probe[nm] = {"padding_frac": 1 - bb.valid_rows.numel() / bb.n_rows,
                             "bitwise_identical": all(torch.equal(outs[0], o) for o in outs[1:]),
                             "max_abs_diff": max(float((outs[0] - o).abs().max()) for o in outs[1:])}
            for h in hs:
                h.remove()
            R["determinism"] = probe
        except Exception as ex:  # noqa: BLE001
            R["determinism"] = {"error": f"{type(ex).__name__}: {str(ex)[:300]}"}
        save()
        print("[atlas] determinism", json.dumps(R["determinism"]), flush=True)

        # ---- standalone GEMMs at model shapes
        d, f = info["d_model"], info["d_ff"]
        M = a.tokens_per_batch if cuda else 1024
        shapes = {"ffn_up": (M, d, f), "ffn_down": (M, f, d), "attn_proj": (M, d, d)}
        G = {}
        for nm, (m, k, n) in shapes.items():
            for dtn, dt in (("bf16", torch.bfloat16), ("tf32", torch.float32)):
                if not cuda and dtn == "bf16":
                    continue
                x = torch.randn(m, k, device=dev, dtype=dt)
                w = torch.randn(k, n, device=dev, dtype=dt)
                prev = torch.backends.cuda.matmul.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = dtn == "tf32"
                for _ in range(3):
                    x @ w
                sync(dev)
                reps = 30 if cuda else 3
                g0 = time.time()
                for _ in range(reps):
                    x @ w
                sync(dev)
                g1 = time.time()
                torch.backends.cuda.matmul.allow_tf32 = prev
                G[f"{nm}_{dtn}"] = {"t_start": g0, "t_end": g1, "tflops": 2 * m * k * n * reps / (g1 - g0) / 1e12, "shape": [m, k, n]}
                del x, w
        if cuda:  # sustained bf16 FFN-up GEMM: does the clock hold under continuous load?
            m, k, n = shapes["ffn_up"]
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            w = torch.randn(k, n, device=dev, dtype=torch.bfloat16)
            s0 = time.time()
            reps = 0
            while time.time() - s0 < a.sustain_s:
                for _ in range(20):
                    x @ w
                sync(dev)
                reps += 20
            s1 = time.time()
            G["sustained_ffn_up_bf16"] = {"t_start": s0, "t_end": s1, "tflops": 2 * m * k * n * reps / (s1 - s0) / 1e12}
            del x, w
        R["gemm"] = G
        save()
        print("[atlas] gemm", json.dumps({k: round(v["tflops"], 1) for k, v in G.items()}), flush=True)

        # ---- Triton tile sweep on real activations (early-exit kernel)
        if cuda:
            try:
                from atlas.filter_triton import filter_candidates
                grabbed, cur = [], {}

                def grab(li, H):
                    if li == a.probe_layer:
                        grabbed.append(H.index_select(0, cur["rows"]))
                hs = attach(model, grab)
                for b in batches:
                    cur["rows"] = b.valid_rows.to(dev)
                    fwd(b)
                    if sum(x.shape[0] for x in grabbed) >= 32768:
                        break
                for h in hs:
                    h.remove()
                H = torch.cat(grabbed)[:32768].contiguous()
                del grabbed
                T, D = H.shape
                rowmap = torch.arange(T, device=dev)
                Hf = H.float()
                sweep = {}
                for regime, kth in (("late_~2_per_col", 3), ("early_~30_per_col", 31)):
                    tp = torch.topk(Hf, kth, dim=0).values[-1]
                    tn = torch.topk(-Hf, kth, dim=0).values[-1]
                    n_sv = int(((Hf > tp) | (-Hf > tn)).sum())
                    cap = max(4 * n_sv, 1 << 16)
                    res = {"survivors": n_sv}
                    for bt, bd, nw in [(16, 256, 4), (32, 256, 4), (16, 512, 4), (32, 512, 8), (64, 256, 8),
                                       (8, 512, 4), (8, 1024, 8), (64, 128, 4), (128, 128, 8), (32, 1024, 8)]:
                        try:
                            v, r, c, n = filter_candidates(H, rowmap, tp, tn, cap, BLOCK_T=bt, BLOCK_D=bd, num_warps=nw)
                            assert int(n) == n_sv, (int(n), n_sv)
                            for _ in range(3):
                                filter_candidates(H, rowmap, tp, tn, cap, BLOCK_T=bt, BLOCK_D=bd, num_warps=nw)
                            torch.cuda.synchronize()
                            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            e0.record()
                            for _ in range(20):
                                filter_candidates(H, rowmap, tp, tn, cap, BLOCK_T=bt, BLOCK_D=bd, num_warps=nw)
                            e1.record()
                            torch.cuda.synchronize()
                            res[f"{bt}x{bd}_w{nw}"] = e0.elapsed_time(e1) / 20
                        except Exception as ex:  # noqa: BLE001
                            res[f"{bt}x{bd}_w{nw}"] = f"error: {str(ex)[:120]}"
                    m = torch.zeros(1, device=dev)

                    def torch_path():
                        mk = (H > tp) | (H < -tn)
                        return torch.nonzero(mk)
                    for _ in range(3):
                        torch_path()
                    torch.cuda.synchronize()
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    for _ in range(20):
                        torch_path()
                    e1.record()
                    torch.cuda.synchronize()
                    res["torch_nonzero_ms"] = e0.elapsed_time(e1) / 20
                    del m
                    sweep[regime] = res
                sweep["one_read_ms_at_peak_bw"] = None
                R["tiles"] = sweep
            except Exception as ex:  # noqa: BLE001
                R["tiles"] = {"error": f"{type(ex).__name__}: {str(ex)[:400]}"}
            save()
            print("[atlas] tiles", json.dumps(R["tiles"])[:1500], flush=True)
    print("[atlas] profile_forward done", flush=True)


if __name__ == "__main__":
    main()
