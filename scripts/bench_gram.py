"""Microbench of the subsample-statistics GEMMs and the one-pass read floor, at the per-layer
shape of a 650M batch (n_s ~ 1.6k hashed rows, D = 5120). Answers: what the Gram costs in TF32
on fp32 copies (torch < 2.8 path) vs bf16 tensor cores with fp32 output (torch >= 2.8,
mm(out_dtype=)), what the enrichment GEMM costs in TF32 vs int8, and how long one read of Hs
takes (the floor for any fused one-pass statistics kernel).

  python scripts/bench_gram.py --n 1600 --D 5120 --A 144 --out bench_gram.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch


def timeit(fn, dev, reps=20, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        if dev == "cuda":
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        else:
            t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    return {"median_ms": statistics.median(ts), "min_ms": min(ts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--D", type=int, default=5120)
    ap.add_argument("--A", type=int, default=144)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=dev).manual_seed(0)
    Xb = torch.randn(a.n, a.D, device=dev, generator=g).to(torch.bfloat16)
    Xf = Xb.float()
    ann_f = (torch.rand(a.n, a.A, device=dev, generator=g) < 0.05).float()
    M = (torch.rand(3 * a.D, a.n, device=dev, generator=g) < 0.1)
    res = {"device": torch.cuda.get_device_name() if dev == "cuda" else "cpu", "torch": torch.__version__,
           "shape": {"n": a.n, "D": a.D, "A": a.A}, "ops": {}}
    flops_gram = 2 * a.D * a.D * a.n

    def run(name, fn, flops=None, bytes_=None):
        r = timeit(fn, dev, a.reps)
        if flops:
            r["TFLOPS"] = flops / r["median_ms"] / 1e9
        if bytes_:
            r["GBps"] = bytes_ / r["median_ms"] / 1e6
        res["ops"][name] = r
        print(f"{name:34s} {r['median_ms']:8.3f} ms" + (f"  {r.get('TFLOPS', 0):6.1f} TFLOPS" if flops else "")
              + (f"  {r.get('GBps', 0):6.0f} GB/s" if bytes_ else ""), flush=True)

    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    run("gram_tf32_fp32_inputs", lambda: Xf.T @ Xf, flops_gram)
    run("gram_cast_plus_tf32", lambda: (lambda x: x.T @ x)(Xb.float()), flops_gram)
    torch.backends.cuda.matmul.allow_tf32 = prev
    run("gram_bf16_out_bf16 (proxy)", lambda: Xb.T @ Xb, flops_gram)
    try:
        torch.mm(Xb.T, Xb, out_dtype=torch.float32)
        run("gram_bf16_out_fp32 (out_dtype)", lambda: torch.mm(Xb.T, Xb, out_dtype=torch.float32), flops_gram)
    except (TypeError, RuntimeError, NotImplementedError) as e:
        res["ops"]["gram_bf16_out_fp32 (out_dtype)"] = {"unavailable": str(e)[:160]}
        print(f"gram_bf16_out_fp32 (out_dtype)     unavailable: {str(e)[:100]}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    run("enrich_tf32 [3D,n]@[n,A]", lambda: M.float() @ ann_f)
    torch.backends.cuda.matmul.allow_tf32 = prev
    if dev == "cuda":
        A8 = -(-a.A // 8) * 8
        ann8 = torch.zeros(a.n, A8, dtype=torch.int8, device=dev)
        ann8[:, :a.A] = ann_f.to(torch.int8)
        try:
            run("enrich_int8 _int_mm", lambda: torch._int_mm(M.to(torch.int8), ann8))
        except RuntimeError as e:
            res["ops"]["enrich_int8 _int_mm"] = {"unavailable": str(e)[:160]}
    nbytes = Xb.numel() * 2
    run("read floor: Hs_bf16.sum(0)", lambda: Xb.sum(0, dtype=torch.float32), bytes_=nbytes)
    run("read floor: Hs_fp32.sum(0)", lambda: Xf.sum(0), bytes_=2 * nbytes)
    run("moments v1/v2: sum + sq", lambda: (Xf.sum(0), (Xf * Xf).sum(0)), bytes_=2 * nbytes * 4)
    E = torch.sort(torch.randn(a.D, 6, device=dev, generator=g), dim=1).values
    HT = Xf.T.contiguous()
    run("transpose copy [n,D]->[D,n]", lambda: Xf.T.contiguous(), bytes_=4 * nbytes)
    run("searchsorted [D,6] x [D,n]", lambda: torch.searchsorted(E, HT, right=True), bytes_=2 * nbytes + 4 * nbytes * 2)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
