"""Fused compare-and-compact: one read of H, survivors written to a preallocated buffer.

survivor (t, d)  <=>  rowmap[t] >= 0  and  (h > thr_pos[d]  or  -h > thr_neg[d])

Each program handles a [BLOCK_T, BLOCK_D] tile: 2-D exclusive prefix sum of its hit
mask (within-row cumsum + row offsets), ONE atomic_add to reserve a contiguous output
range, masked stores. Nothing is synced here: the count stays on device and is read
by Atlas.end_batch together with every other layer's count.

Works under TRITON_INTERPRET=1 on CPU for testing (slow).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _filter_kernel(
    H_ptr, rowmap_ptr, thr_pos_ptr, thr_neg_ptr,
    out_val_ptr, out_row_ptr, out_col_ptr, counter_ptr,
    T, D, stride_t, MAX_OUT,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    t_off = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    t_ok = t_off < T
    d_ok = d_off < D

    vrow = tl.load(rowmap_ptr + t_off, mask=t_ok, other=-1)          # int64, -1 = padding
    thr_p = tl.load(thr_pos_ptr + d_off, mask=d_ok, other=float("inf"))
    thr_n = tl.load(thr_neg_ptr + d_off, mask=d_ok, other=float("inf"))

    tile_ok = t_ok[:, None] & d_ok[None, :]
    ptrs = H_ptr + t_off.to(tl.int64)[:, None] * stride_t + d_off[None, :]
    h = tl.load(ptrs, mask=tile_ok, other=0.0).to(tl.float32)

    hit = ((h > thr_p[None, :]) | (-h > thr_n[None, :])) & (vrow[:, None] >= 0) & tile_ok
    hit_i = hit.to(tl.int32)

    row_cnt = tl.sum(hit_i, axis=1)                    # [BLOCK_T]
    n_hit = tl.sum(row_cnt, axis=0)
    if n_hit > 0:  # most tiles have no survivors once tau has risen: skip scan, atomic and stores
        row_pre = tl.cumsum(row_cnt, axis=0) - row_cnt     # exclusive over rows
        in_row = tl.cumsum(hit_i, axis=1) - hit_i          # exclusive within row
        base = tl.atomic_add(counter_ptr, n_hit)           # one atomic per non-empty tile
        slot = base + row_pre[:, None] + in_row
        ok = hit & (slot < MAX_OUT)
        r2 = tl.broadcast_to(vrow.to(tl.int32)[:, None], (BLOCK_T, BLOCK_D))
        d2 = tl.broadcast_to(d_off[None, :], (BLOCK_T, BLOCK_D))
        tl.store(out_val_ptr + slot, h, mask=ok)
        tl.store(out_row_ptr + slot, r2, mask=ok)
        tl.store(out_col_ptr + slot, d2, mask=ok)


_BUF: dict = {}


DEFAULT_TILE = {"BLOCK_T": 16, "BLOCK_D": 256, "num_warps": 4}  # phase-1 sweep best (pre early-exit); retuned on GPU


def filter_candidates(H, rowmap, thr_pos, thr_neg, max_out, BLOCK_T=None, BLOCK_D=None, num_warps=None):
    """Returns (vals fp32 [max_out], rows int64 [max_out], cols int64 [max_out], count (0-d
    device tensor)). Only the first min(count, max_out) entries are meaningful; count >
    max_out means overflow (caller redoes the batch densely for this layer)."""
    assert H.stride(1) == 1
    BLOCK_T = BLOCK_T or DEFAULT_TILE["BLOCK_T"]
    BLOCK_D = BLOCK_D or DEFAULT_TILE["BLOCK_D"]
    num_warps = num_warps or DEFAULT_TILE["num_warps"]
    T, D = H.shape
    dev = H.device
    out_val = torch.empty(max_out, device=dev, dtype=torch.float32)
    out_row = torch.empty(max_out, device=dev, dtype=torch.int32)
    out_col = torch.empty(max_out, device=dev, dtype=torch.int32)
    counter = torch.zeros(1, device=dev, dtype=torch.int32)
    grid = (triton.cdiv(T, BLOCK_T), triton.cdiv(D, BLOCK_D))
    _filter_kernel[grid](
        H, rowmap, thr_pos.contiguous().float(), thr_neg.contiguous().float(),
        out_val, out_row, out_col, counter,
        T, D, H.stride(0), max_out,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D, num_warps=num_warps,
    )
    return out_val, out_row.long(), out_col.long(), counter[0]
