"""Triton compaction kernel == torch reference. On CPU: TRITON_INTERPRET=1 python -m tests.test_triton_kernel
On GPU: python -m tests.test_triton_kernel --cuda   (also runs the end-to-end Atlas check with backend=triton)."""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, ".")


def check_kernel(dev, dtype, T=777, D=300, frac_pad=0.15, seed=0, BLOCK_T=32, BLOCK_D=128):
    from atlas.filter_triton import filter_candidates

    g = torch.Generator().manual_seed(seed)
    H = (torch.randn(T, D, generator=g) * 2).to(dtype).to(dev)
    valid = torch.rand(T, generator=g) > frac_pad
    rowmap = torch.full((T,), -1, dtype=torch.long)
    rowmap[valid] = torch.arange(int(valid.sum()))
    rowmap = rowmap.to(dev)
    thr_p = (torch.rand(D, generator=g) * 2 + 2.5).to(dev)
    thr_n = (torch.rand(D, generator=g) * 2 + 2.5).to(dev)
    thr_n[:7] = float("-inf")  # columns where everything valid survives (warm-up-like)
    cap = 1 << 16
    v, r, c, cnt = filter_candidates(H, rowmap, thr_p, thr_n, cap, BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D)
    n = int(cnt)
    Hf = H.float()
    ref = ((Hf > thr_p) | (-Hf > thr_n)) & (rowmap >= 0)[:, None]
    rt, rd = torch.nonzero(ref, as_tuple=True)
    got = set(zip(r[:n].tolist(), c[:n].tolist()))
    exp = set(zip(rowmap[rt].tolist(), rd.tolist()))
    assert n == len(exp), (n, len(exp))
    assert got == exp, f"{len(got - exp)} extra, {len(exp - got)} missing"
    inv = torch.full((int(rowmap.max()) + 1,), -1, dtype=torch.long, device=dev)
    inv[rowmap[rowmap >= 0]] = torch.nonzero(rowmap >= 0).squeeze(1)
    assert torch.equal(v[:n], Hf[inv[r[:n]], c[:n]]), "survivor values wrong"
    # overflow reporting: count must exceed a too-small cap, and no out-of-bounds writes
    v2, r2, c2, cnt2 = filter_candidates(H, rowmap, thr_p, thr_n, 64, BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D)
    assert int(cnt2) == n and n > 64
    print(f"kernel OK on {dev} {dtype}: T={T} D={D} (non-multiples of the {BLOCK_T}x{BLOCK_D} tile), "
          f"{n} survivors incl. -inf columns, overflow detected (count {int(cnt2)} > cap 64)")


if __name__ == "__main__":
    cuda = "--cuda" in sys.argv
    dev = "cuda" if cuda else "cpu"
    if not cuda:
        assert os.environ.get("TRITON_INTERPRET") == "1", "set TRITON_INTERPRET=1 for CPU"
    for dtype in (torch.float32, torch.bfloat16):
        try:
            check_kernel(dev, dtype)
        except Exception as e:  # noqa: BLE001
            print(f"kernel FAILED on {dev} {dtype}: {type(e).__name__}: {str(e)[:300]}")
            if cuda:
                raise
    if cuda:
        import tests.test_exactness as T
        ok = T.run(torch.bfloat16, backend="triton", device="cuda", verbose=False)
        ok &= T.run(torch.bfloat16, backend="torch", device="cuda", verbose=False)
        sys.exit(0 if ok else 1)
