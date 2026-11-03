"""Streamed Atlas == offline reference, on synthetic multi-layer data with padding,
variable batch sizes, heavy tails and a deliberate distribution shift (forces overflow ->
dense redo). Run: python -m tests.test_exactness  (from repo root). Exit code != 0 on failure.
"""
from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, ".")
from atlas.core import Atlas, AtlasConfig, pack_tok  # noqa: E402
from atlas.reference import pilot_edges, reference  # noqa: E402


def make_stream(L, D, n_seq, seed, dtype):
    g = torch.Generator().manual_seed(seed)
    lens = torch.randint(5, 120, (n_seq,), generator=g)
    scale = torch.rand(L, D, generator=g) * 3 + 0.1
    shift = torch.randn(L, D, generator=g)
    seqs = []
    for i, n in enumerate(lens.tolist()):
        H = torch.randn(L, n, D, generator=g) * scale[:, None] + shift[:, None]
        H = H * (1 + 4 * (torch.rand(L, n, D, generator=g) < 0.002))       # outlier spikes
        if i > 0.7 * n_seq:                                                    # late distribution shift
            H = H + 2.5 * scale[:, None]
        seqs.append(H.to(dtype))
    return lens, seqs


def run(dtype, backend="torch", n_seq=1200, L=3, D=96, K=8, Kr=4, n_annot=7, seed=0, verbose=True, device="cpu",
        seeded=False, seed_c=4.0):
    lens, seqs = make_stream(L, D, n_seq, seed, dtype)
    g = torch.Generator().manual_seed(seed + 1)
    ann_full = [torch.randint(-1, n_annot, (n, 3), generator=g) for n in lens.tolist()]
    # pilot from the first 150 sequences (edges then fixed)
    Hp = [torch.cat([s[l] for s in seqs[:150]]) for l in range(L)]
    E = pilot_edges(Hp)
    cfg = AtlasConfig(K=K, Kr=Kr, rho_res=0.05, rho_sub=0.2, cap_frac=0.05, backend=backend, seed=7)
    atlas = Atlas(L, D, E, cfg, n_annot=n_annot, device=device)
    if seeded:  # pilot tokens (first 150 seqs) are streamed below, as in run_atlas
        from atlas.reference import pilot_seeds
        ts, rk = pilot_seeds(Hp, E, K, Kr, n_total=int(lens.sum()), c=seed_c)
        atlas.seed(ts, rk)
    t0 = time.perf_counter()
    i = 0
    bi = 0
    while i < n_seq:
        bsz = int(torch.randint(4, 40, (1,), generator=g))
        idx = list(range(i, min(n_seq, i + bsz)))
        i += len(idx)
        Lmax = max(lens[j] for j in idx) + 2                                    # BOS/EOS padding layout
        T = len(idx) * Lmax
        Hb = [torch.randn(T, D).to(dtype) * 50 for _ in range(L)]               # garbage at padding
        valid_rows, tok, ann = [], [], []
        for r, j in enumerate(idx):
            n = int(lens[j])
            rows = r * Lmax + 1 + torch.arange(n)
            for l in range(L):
                Hb[l][rows] = seqs[j][l]
            valid_rows.append(rows)
            tok.append(pack_tok(torch.full((n,), j), torch.arange(n)))
            ann.append(ann_full[j])
        valid_rows, tok, ann = torch.cat(valid_rows), torch.cat(tok), torch.cat(ann)
        ctx = atlas.make_ctx(T, valid_rows, tok, ann)

        Hb = [h.to(device) for h in Hb]

        def forward(Hb=Hb):
            for l in range(L):
                atlas.observe(l, Hb[l])

        atlas.begin_batch(ctx)
        forward()
        redo = atlas.end_batch()
        if redo:
            atlas.begin_batch(ctx, redo_layers=redo)
            forward()
            assert atlas.end_batch() == []
        bi += 1
    dt = time.perf_counter() - t0
    got = atlas.result()

    Hall = [torch.cat([s[l] for s in seqs]) for l in range(L)]
    toks = torch.cat([pack_tok(torch.full((int(n),), j), torch.arange(int(n))) for j, n in enumerate(lens)])
    annf = torch.zeros(toks.numel(), n_annot)
    A = torch.cat(ann_full)
    r, c = torch.nonzero(A >= 0, as_tuple=True)
    annf[r, A[r, c]] = 1
    ref = reference(Hall, toks, E, K, Kr, cfg.rho_sub, cfg.seed, ann=annf, levels=cfg.enrich_levels)

    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        if verbose or not cond:
            print(f"  {'PASS' if cond else 'FAIL'} {name}")

    check("top-K values exact", torch.equal(got["top_val"], ref["top_val"]))
    offs = torch.cumsum(lens, 0) - lens
    def rows_of(t):
        return offs[t // 1024] + t % 1024
    # every stored top-K token must attain its stored value (catches payload bugs)
    att = True
    for l in range(L):
        H = Hall[l].float()
        tt = got["top_tok"][l]                                   # [2, D, K]
        cols = torch.arange(D)[None, :, None].expand_as(tt)
        att &= bool((H[rows_of(tt), cols] == got["top_val"][l]).all())
    check("top-K tokens attain their stored values", att)
    differ = got["top_tok"] != ref["top_tok"]
    if dtype == torch.float32:
        check("top-K token ids exact", not differ.any())
    else:
        print(f"  info: {int(differ.sum())} top-K id differences, all exact ties (values equal, ids attain them)")
    att = True
    for l in range(L):
        H = Hall[l].float()
        rt = got["res_tok"][l]
        m = rt >= 0
        cols = torch.arange(D)[None, :, None, None].expand_as(rt)
        att &= bool((H[rows_of(rt[m]), cols[m]] == got["res_val"][l][m]).all())
    check("reservoir tokens attain their stored values", att)
    lim = got["res_seed_limited"].unsqueeze(-1).expand_as(got["res_tok"])  # bins verification rejects
    ok_bins = ~lim
    if seeded:
        print(f"  info: seeds active on {got['res_seeded_bins']} bins; verification rejects {got['res_seed_limited_bins']}; "
              f"top-K groups short of K under seed: {got['top_seed_short_groups']}")
    check("top-K: no group short under its seed", got["top_seed_short_groups"] == 0)
    rk_got = got["res_key"].where(got["res_tok"] >= 0, torch.iinfo(torch.int64).max)
    check("reservoir keys exact (verified bins)", torch.equal(rk_got[ok_bins], ref["res_key"][ok_bins]))
    check("reservoir tokens exact (verified bins)", torch.equal(got["res_tok"][ok_bins], ref["res_tok"][ok_bins]))
    check("reservoir values exact (verified bins)", torch.allclose(got["res_val"][ok_bins], ref["res_val"][ok_bins], equal_nan=True, atol=0, rtol=0))
    if seeded and got["res_seed_limited_bins"]:
        # a rejected bin must hold exactly the bin's keys below its seed (all of them, nothing else).
        # ref holds the Kr smallest keys of the bin; when fewer than Kr are under the seed they are all there.
        MAX = torch.iinfo(torch.int64).max
        sk = got["res_seed_key"].unsqueeze(-1)
        want = torch.where(ref["res_key"] < sk, ref["res_key"], torch.full_like(ref["res_key"], MAX)).sort(-1).values
        have = rk_got.sort(-1).values
        m = got["res_seed_limited"]
        check(f"rejected bins hold exactly the bin's keys under the seed ({int(m.sum())} bins)", torch.equal(have[m], want[m]))
    check("n_sub", got["n_sub"] == ref["n_sub"])
    check("subsample bin masses exact", torch.equal(got["sub_bin"].double(), ref["sub_bin"]))
    check("subsample sums", torch.allclose(got["sub_sum"], ref["sub_sum"], rtol=1e-6, atol=1e-6))
    check("subsample sum of squares", torch.allclose(got["sub_sq"], ref["sub_sq"], rtol=1e-6, atol=1e-6))
    check("enrichment counts exact", torch.equal(got["enr"].double(), ref["enr"].double()))
    check("Gram (fp32 accum) rel err < 1e-5", ((got["gram"] - ref["gram"]).abs().max() / ref["gram"].abs().max()) < 1e-5)
    s = got["stats"]
    N = int(lens.sum())
    frac = s["compacted"] / max(1, s["elements"])
    print(f"[{dtype}, {backend}, {device}] {'OK' if ok else 'FAILED'}  N={N} tokens, L={L}, D={D}, K={K}, Kr={Kr}, "
          f"{bi} batches, {dt:.2f}s; dense layer-batches={s['dense_layer_batches']}, redo layers={s['redo_layers']}; "
          f"compacted fraction {frac:.2e}")
    return ok


if __name__ == "__main__":
    ok = run(torch.float32) & run(torch.bfloat16)
    print("-- seeded (top-K from pilot 2K-th, reservoir c=4)")
    ok &= run(torch.bfloat16, seeded=True, verbose=False)
    print("-- seeded too tight (c=0.25): verification must flag bins")
    ok &= run(torch.bfloat16, seeded=True, seed_c=0.25, verbose=False)
    sys.exit(0 if ok else 1)
