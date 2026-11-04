"""Triton subsample-statistics and Gram kernels vs torch references.

CPU (interpreter, small tiles):  TRITON_INTERPRET=1 python -m tests.test_stats_triton
GPU (default tiles, 650M shape, timings):  python -m tests.test_stats_triton --cuda
"""
from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, ".")
from atlas.core import bin_of  # noqa: E402
from atlas.stats_triton import gram_accumulate, sub_stats, symmetrize_upper  # noqa: E402


def case(n, D, B, dtype, dev, levels=(2, 3, 4), small=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = (torch.randn(n, D, generator=g) * (torch.rand(D, generator=g) * 2 + 0.2)).to(dtype)
    Xf = X.float()
    q = torch.tensor([0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999][:B])
    k = (q * (n - 1)).long()
    E0 = Xf.sort(0).values[k].T.contiguous()                 # edges are data values: exact ties occur
    E1 = (-Xf).sort(0).values[k].T.contiguous()
    kw = {"BLOCK_N": 8, "BLOCK_D": 32} if small else {}
    t0 = time.perf_counter()
    s, sq, cnt, M = sub_stats(X.to(dev), E0.to(dev), E1.to(dev), levels, **kw)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    s, sq, cnt, M = s.cpu(), sq.cpu(), cnt.cpu(), M.cpu()
    ok = True
    b0, b1 = bin_of(Xf, E0), bin_of(-Xf, E1)
    for p, b in ((0, b0), (1, b1)):
        want = torch.stack([(b == bb).sum(0) for bb in range(B)], 1)
        ok &= torch.equal(cnt[p].long(), want)
        n8 = M.shape[-1]
        for j, lev in enumerate(levels):
            m = torch.zeros(D, n8, dtype=torch.int8)
            m[:, :n] = (b >= lev).T.to(torch.int8)
            ok &= torch.equal(M[p, j], m)
    ok &= torch.allclose(s.double(), Xf.double().sum(0), rtol=1e-5, atol=1e-4)
    ok &= torch.allclose(sq.double(), (Xf.double() ** 2).sum(0), rtol=1e-5, atol=1e-4)
    ties = int((Xf[:, :, None] == E0[None]).sum())
    print(f"sub_stats {'OK' if ok else 'FAILED'}: n={n} D={D} B={B} {dtype} on {dev}, {ties} values equal to an edge, {dt * 1e3:.2f} ms", flush=True)
    return ok


def gram_case(n, D, dtype, dev, small=True, seed=1):
    g = torch.Generator().manual_seed(seed)
    X1 = torch.randn(n, D, generator=g).to(dtype)
    X2 = torch.randn(n // 2 + 3, D, generator=g).to(dtype)
    C = torch.zeros(D, D, device=dev)
    kw = {"BM": 32, "BN": 32, "BK": 16, "num_warps": 4, "num_stages": 1} if small else {}
    t0 = time.perf_counter()
    gram_accumulate(X1.to(dev), C, **kw)
    gram_accumulate(X2.to(dev), C, **kw)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    G = symmetrize_upper(C.cpu().double())
    R = X1.double().T @ X1.double() + X2.double().T @ X2.double()
    rel = float((G - R).abs().max() / R.abs().max())
    BM = kw.get("BM", 128)
    nb = -(-D // BM)
    low = torch.zeros(D, D, dtype=torch.bool)
    for i in range(nb):
        for j in range(i):
            low[i * BM:(i + 1) * BM, j * BM:(j + 1) * BM] = True
    untouched = bool((C.cpu()[low] == 0).all())
    ok = rel < 1e-5 and untouched
    print(f"gram {'OK' if ok else 'FAILED'}: n={n} D={D} {dtype} on {dev}, rel err {rel:.2e} vs fp64, "
          f"lower blocks untouched {untouched}, 2 accumulations {dt * 1e3:.2f} ms", flush=True)
    return ok


def main():
    cuda = "--cuda" in sys.argv
    ok = True
    if not cuda:
        for (n, D, B, dt) in [(203, 150, 5, torch.bfloat16), (64, 70, 6, torch.float32), (5, 33, 5, torch.bfloat16)]:
            ok &= case(n, D, B, dt, "cpu")
        ok &= case(37, 40, 5, torch.bfloat16, "cpu", levels=())
        # the interpreter's bf16 tl.dot is broken (bf16 loads round-trip exactly; fp16, same code path, is
        # right), so half precision is checked with fp16 here and bf16 on the GPU (--cuda)
        ok &= gram_case(50, 100, torch.float16, "cpu")
        ok &= gram_case(19, 64, torch.float32, "cpu")
    else:
        for (n, D, B, dt) in [(203, 150, 5, torch.bfloat16), (1600, 5120, 6, torch.bfloat16), (1611, 5120, 5, torch.float32)]:
            ok &= case(n, D, B, dt, "cuda", small=False)
            ok &= case(n, D, B, dt, "cuda", small=False)  # second call: no compile time
        ok &= gram_case(1600, 5120, torch.bfloat16, "cuda", small=False)
        ok &= gram_case(1600, 5120, torch.bfloat16, "cuda", small=False)
        ok &= gram_case(301, 1000, torch.float32, "cuda", small=False)
    print("ALL OK" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
