"""Hashed-subsample statistics in two Triton kernels (stats_impl "v3").

_sub_stats_kernel: one read of the gathered subsample rows X [n, D]. Each program owns a strip
  of BLOCK_D columns and walks all rows, keeping per-column accumulators in registers: bin
  counts for both poles (bin(s) = #{b : s >= E_b} - 1, s = +h / -h), sum and sum of squares.
  One store per column at the end (no atomics: deterministic). It also writes, for every
  enrichment level, the 0/1 mask (bin >= level) as int8 in the layout the exact int8 GEMM
  wants: masks[p, j, d, row], rows padded to a multiple of 8 with zeros. Nothing [n, D, B]
  or int64 is materialised (the 2026-11-04 A/B showed those passes cost as much as the work).

_gram_kernel: C[D, D] (fp32) += X^T X for the upper block triangle only (C's lower blocks stay
  zero; Atlas symmetrises once at the end). bf16 inputs on tensor cores with fp32 accumulation
  (products of bf16 values are exact in fp32), accumulated into C in place: no separate
  read-modify-write pass, half the FLOPs of the full product.

Both run under TRITON_INTERPRET=1 on CPU for testing (slow).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sub_stats_kernel(X_ptr, n, n8, D, stride_x, chunk,
                      E0_ptr, E1_ptr, lev_ptr,
                      sum_ptr, sq_ptr, cnt_ptr, mask_ptr,
                      B: tl.constexpr, BP: tl.constexpr, NL: tl.constexpr,
                      BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    pr = tl.program_id(1)                     # row split: program pr owns rows [pr*chunk, (pr+1)*chunk)
    r_lo = pr * chunk
    r_hi = tl.minimum(r_lo + chunk, n8)
    d = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    dm = d < D
    bi = tl.arange(0, BP)
    bm = bi < B
    em = dm[:, None] & bm[None, :]
    e0 = tl.load(E0_ptr + d[:, None] * B + bi[None, :], mask=em, other=float("inf"))   # [BLOCK_D, BP]
    e1 = tl.load(E1_ptr + d[:, None] * B + bi[None, :], mask=em, other=float("inf"))
    acc_s = tl.zeros([BLOCK_D], dtype=tl.float64)   # fp32 within a row block, fp64 across blocks
    acc_q = tl.zeros([BLOCK_D], dtype=tl.float64)
    c0 = tl.zeros([BLOCK_D, BP], dtype=tl.int32)
    c1 = tl.zeros([BLOCK_D, BP], dtype=tl.int32)
    for r0 in tl.range(r_lo, r_hi, BLOCK_N):
        r = r0 + tl.arange(0, BLOCK_N)
        rm = r < n
        valid = rm[:, None] & dm[None, :]
        x = tl.load(X_ptr + r.to(tl.int64)[:, None] * stride_x + d[None, :], mask=valid, other=0.0).to(tl.float32)
        acc_s += tl.sum(x, axis=0).to(tl.float64)
        acc_q += tl.sum(x * x, axis=0).to(tl.float64)
        b0 = tl.sum((x[:, :, None] >= e0[None, :, :]).to(tl.int32), axis=2) - 1           # [BLOCK_N, BLOCK_D]
        b1 = tl.sum((-x[:, :, None] >= e1[None, :, :]).to(tl.int32), axis=2) - 1
        b0 = tl.where(valid, b0, -1)
        b1 = tl.where(valid, b1, -1)
        c0 += tl.sum((b0[:, :, None] == bi[None, None, :]).to(tl.int32), axis=0)
        c1 += tl.sum((b1[:, :, None] == bi[None, None, :]).to(tl.int32), axis=0)
        st = (r < r_hi)[:, None] & dm[None, :]
        for j in tl.static_range(NL):
            lev = tl.load(lev_ptr + j)
            off0 = (j * D + d[None, :]).to(tl.int64) * n8 + r[:, None]
            off1 = ((NL + j) * D + d[None, :]).to(tl.int64) * n8 + r[:, None]
            tl.store(mask_ptr + off0, (b0 >= lev).to(tl.int8), mask=st)
            tl.store(mask_ptr + off1, (b1 >= lev).to(tl.int8), mask=st)
    tl.store(sum_ptr + pr * D + d, acc_s, mask=dm)             # partials [R, ...], reduced in a fixed order
    tl.store(sq_ptr + pr * D + d, acc_q, mask=dm)
    tl.store(cnt_ptr + (pr * 2 * D + d[:, None]) * B + bi[None, :], c0, mask=em)
    tl.store(cnt_ptr + (pr * 2 * D + D + d[:, None]) * B + bi[None, :], c1, mask=em)


STATS_TILE = {"BLOCK_N": 16, "BLOCK_D": 64, "num_warps": 4, "R": 1}  # R=0: enough row splits for ~4 programs per SM
_LEV: dict = {}
_NSM: dict = {}


def sub_stats(X: torch.Tensor, E0: torch.Tensor, E1: torch.Tensor, levels, BLOCK_N=None, BLOCK_D=None, num_warps=None, R=None):
    """X [n, D] (bf16/fp32, row stride arbitrary, unit column stride); E0, E1 [D, B] fp32 edges
    for s = +h and s = -h. Returns (sum fp64 [D], sumsq fp64 [D], counts int32 [2, D, B],
    masks int8 [2, NL, D, n8] with n8 = ceil(n / 8) * 8, padding rows zero)."""
    assert X.stride(1) == 1
    BLOCK_N = BLOCK_N or STATS_TILE["BLOCK_N"]
    BLOCK_D = BLOCK_D or STATS_TILE["BLOCK_D"]
    num_warps = num_warps or STATS_TILE["num_warps"]
    R = STATS_TILE["R"] if R is None else R
    n, D = X.shape
    B = E0.shape[1]
    NL = len(levels)
    n8 = -(-n // 8) * 8
    dev = X.device
    nblk = max(1, triton.cdiv(n8, BLOCK_N))
    if R == 0:
        if dev not in _NSM:
            _NSM[dev] = torch.cuda.get_device_properties(dev).multi_processor_count if dev.type == "cuda" else 8
        R = max(1, min(nblk, triton.cdiv(4 * _NSM[dev], triton.cdiv(D, BLOCK_D))))
    R = max(1, min(R, nblk))
    chunk = triton.cdiv(nblk, R) * BLOCK_N
    R = triton.cdiv(max(n8, 1), chunk)
    osum = torch.empty(R, D, device=dev, dtype=torch.float64)
    osq = torch.empty(R, D, device=dev, dtype=torch.float64)
    ocnt = torch.empty(R, 2, D, B, device=dev, dtype=torch.int32)
    omask = torch.empty(2, max(NL, 1), D, max(n8, 8), device=dev, dtype=torch.int8)
    lk = (tuple(levels), dev)
    if lk not in _LEV:  # cached: no host-to-device copy per call
        _LEV[lk] = torch.tensor(list(levels) or [0], device=dev, dtype=torch.int32)
    lev = _LEV[lk]
    grid = (triton.cdiv(D, BLOCK_D), R)
    _sub_stats_kernel[grid](X, n, n8, D, X.stride(0), chunk,
                            E0.contiguous().float(), E1.contiguous().float(), lev,
                            osum, osq, ocnt, omask,
                            B=B, BP=triton.next_power_of_2(max(B, 2)), NL=NL,
                            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, num_warps=num_warps)
    if R > 1:
        osum, osq, ocnt = osum.sum(0), osq.sum(0), ocnt.sum(0, dtype=torch.int32)
    else:
        osum, osq, ocnt = osum[0], osq[0], ocnt[0]
    return osum, osq, ocnt, omask[:, :NL, :, :n8]


@triton.jit
def _gram_kernel(X_ptr, C_ptr, n, D, stride_x, tm_ptr, tn_ptr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, IEEE: tl.constexpr):
    pid = tl.program_id(0)
    bm = tl.load(tm_ptr + pid)
    bn = tl.load(tn_ptr + pid)
    rm = bm * BM + tl.arange(0, BM)
    rn = bn * BN + tl.arange(0, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in tl.range(0, n, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < n
        a = tl.load(X_ptr + rk.to(tl.int64)[:, None] * stride_x + rm[None, :], mask=km[:, None] & (rm < D)[None, :], other=0.0)
        b = tl.load(X_ptr + rk.to(tl.int64)[:, None] * stride_x + rn[None, :], mask=km[:, None] & (rn < D)[None, :], other=0.0)
        if IEEE:
            acc += tl.dot(tl.trans(a), b, input_precision="ieee")
        else:
            acc += tl.dot(tl.trans(a), b)
    cm = (rm < D)[:, None] & (rn < D)[None, :]
    cp = C_ptr + rm.to(tl.int64)[:, None] * D + rn[None, :]
    c = tl.load(cp, mask=cm, other=0.0)
    tl.store(cp, c + acc, mask=cm)


GRAM_TILE = {"BM": 128, "BN": 128, "BK": 32, "num_warps": 8, "num_stages": 3}
_TRI: dict = {}


def gram_accumulate(X: torch.Tensor, C: torch.Tensor, BM=None, BN=None, BK=None, num_warps=None, num_stages=None):
    """C [D, D] fp32 contiguous += X^T X on the upper block triangle (blocks bm <= bn), in place.
    Symmetrise with symmetrize_upper() before use."""
    assert X.stride(1) == 1 and C.is_contiguous() and C.dtype == torch.float32
    BM = BM or GRAM_TILE["BM"]
    BN = BN or GRAM_TILE["BN"]
    BK = BK or GRAM_TILE["BK"]
    assert BM == BN, "upper block triangle needs square tiles"
    num_warps = num_warps or GRAM_TILE["num_warps"]
    num_stages = num_stages or GRAM_TILE["num_stages"]
    n, D = X.shape
    if n == 0:
        return C
    key = (D, BM, C.device)
    if key not in _TRI:
        nb = triton.cdiv(D, BM)
        tm, tn = torch.triu_indices(nb, nb)
        _TRI[key] = (tm.to(C.device, torch.int32), tn.to(C.device, torch.int32))
    tm, tn = _TRI[key]
    _gram_kernel[(tm.numel(),)](X, C, n, D, X.stride(0), tm, tn, BM=BM, BN=BN, BK=BK,
                                IEEE=X.dtype == torch.float32, num_warps=num_warps, num_stages=num_stages)
    return C


from .core import symmetrize_upper  # noqa: E402,F401  (pure torch; re-exported for tests)
