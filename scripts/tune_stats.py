"""Tile/split sweep for the stats-v3 kernels at the per-layer 650M shape (n_s ~1600 subsample rows,
D = 5120), timed with CUDA events on device-resident inputs (no host copies in the timing). Also
times the alternatives for the Gram: cuBLAS TF32 addmm_ into the fp32 accumulator (full product,
fp32 copies) and a bf16 GEMM with bf16 output (speed ceiling only, not exact enough to use).

  python scripts/tune_stats.py --out tune_stats.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.stats_triton import gram_accumulate, sub_stats  # noqa: E402


def timeit(fn, reps=20, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--D", type=int, default=5120)
    ap.add_argument("--B", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rows-only", action="store_true", help="sweep only the row split R of the stats kernel (default tiles), no Gram")
    a = ap.parse_args()
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    X = torch.randn(a.n, a.D, device=dev, generator=g).to(torch.bfloat16)
    q = torch.tensor([0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999][: a.B], device=dev)
    E0 = X.float().sort(0).values[(q * (a.n - 1)).long()].T.contiguous()
    E1 = (-X.float()).sort(0).values[(q * (a.n - 1)).long()].T.contiguous()
    res = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "shape": vars(a), "stats": [], "gram": []}
    ref = sub_stats(X, E0, E1, (2, 3, 4))
    grid = [(16, 64, R, 4) for R in (1, 2, 4, 6, 8, 16, 0)] if a.rows_only else \
        itertools.product((8, 16, 32), (16, 32, 64), (1, 2, 4, 8, 0), (2, 4, 8))
    for BN, BD, R, nw in grid:
        try:
            out = sub_stats(X, E0, E1, (2, 3, 4), BLOCK_N=BN, BLOCK_D=BD, R=R, num_warps=nw)
            same = torch.equal(out[2], ref[2]) and torch.equal(out[3], ref[3])
            ms = timeit(lambda: sub_stats(X, E0, E1, (2, 3, 4), BLOCK_N=BN, BLOCK_D=BD, R=R, num_warps=nw))
            res["stats"].append({"BLOCK_N": BN, "BLOCK_D": BD, "R": R, "num_warps": nw, "ms": ms, "same": same})
        except Exception as e:  # noqa: BLE001  (configs that do not compile or exceed resources)
            res["stats"].append({"BLOCK_N": BN, "BLOCK_D": BD, "R": R, "num_warps": nw, "error": str(e)[:120]})
    ok = sorted([r for r in res["stats"] if "ms" in r and r["same"]], key=lambda r: r["ms"])
    print("stats kernel, best 5:", json.dumps(ok[:5]), flush=True)
    print("stats kernel, by R (default tiles):", [(r["R"], round(r["ms"], 3)) for r in res["stats"] if "ms" in r and (r["BLOCK_N"], r["BLOCK_D"], r["num_warps"]) == (16, 64, 4)], flush=True)
    if a.rows_only:
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1))
        return
    C = torch.zeros(a.D, a.D, device=dev)
    for BM, BK, nw, ns in itertools.product((64, 128), (32, 64), (4, 8), (2, 3, 4, 5)):
        try:
            C.zero_()
            gram_accumulate(X, C, BM=BM, BN=BM, BK=BK, num_warps=nw, num_stages=ns)
            ms = timeit(lambda: gram_accumulate(X, C, BM=BM, BN=BM, BK=BK, num_warps=nw, num_stages=ns))
            res["gram"].append({"BM": BM, "BK": BK, "num_warps": nw, "num_stages": ns, "ms": ms,
                                "TFLOPS_upper": a.D * a.D * a.n / ms / 1e9})
        except Exception as e:  # noqa: BLE001
            res["gram"].append({"BM": BM, "BK": BK, "num_warps": nw, "num_stages": ns, "error": str(e)[:120]})
    Xf = X.float()
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    res["gram_cublas_tf32_addmm_full_ms"] = timeit(lambda: C.addmm_(Xf.T, Xf))
    res["gram_cublas_tf32_addmm_full_incl_cast_ms"] = timeit(lambda: (lambda x: C.addmm_(x.T, x))(X.float()))
    torch.backends.cuda.matmul.allow_tf32 = prev
    res["gram_bf16_out_bf16_ceiling_ms"] = timeit(lambda: X.T @ X)
    ok = sorted([r for r in res["gram"] if "ms" in r], key=lambda r: r["ms"])
    print("gram kernel, best 5:", json.dumps(ok[:5]), flush=True)
    print("gram alternatives (ms): cuBLAS TF32 addmm full %.3f (incl. cast %.3f), bf16 ceiling %.3f" % (
        res["gram_cublas_tf32_addmm_full_ms"], res["gram_cublas_tf32_addmm_full_incl_cast_ms"], res["gram_bf16_out_bf16_ceiling_ms"]), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
