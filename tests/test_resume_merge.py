"""(1) merge: two Atlases over disjoint halves of a stream, merged == offline reference (exact).
(2) resume: run_atlas on real ESM-2 8M, full run == run stopped between checkpoints + --resume.
python -m tests.test_resume_merge [--resume-data data/swissprot_test500.pt]"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, ".")
from atlas.core import Atlas, AtlasConfig, pack_tok  # noqa: E402
from atlas.reference import pilot_edges, reference  # noqa: E402
from tests.test_exactness import make_stream  # noqa: E402


def feed(atlas, seqs, lens, ann_full, idxs, D, L, dtype, g):
    i = 0
    while i < len(idxs):
        bsz = int(torch.randint(4, 40, (1,), generator=g))
        idx = idxs[i:i + bsz]
        i += len(idx)
        Lmax = max(int(lens[j]) for j in idx) + 2
        T = len(idx) * Lmax
        Hb = [torch.randn(T, D).to(dtype) * 50 for _ in range(L)]
        vr, tk, an = [], [], []
        for r, j in enumerate(idx):
            n = int(lens[j])
            rows = r * Lmax + 1 + torch.arange(n)
            for l in range(L):
                Hb[l][rows] = seqs[j][l]
            vr.append(rows); tk.append(pack_tok(torch.full((n,), j), torch.arange(n))); an.append(ann_full[j])
        ctx = atlas.make_ctx(T, torch.cat(vr), torch.cat(tk), torch.cat(an))
        atlas.begin_batch(ctx)
        for l in range(L):
            atlas.observe(l, Hb[l])
        redo = atlas.end_batch()
        if redo:
            atlas.begin_batch(ctx, redo_layers=redo)
            for l in range(L):
                atlas.observe(l, Hb[l])
            assert atlas.end_batch() == []


def test_merge(dtype=torch.bfloat16, n_seq=1000, L=3, D=64, K=8, Kr=4, n_annot=5, seeded=False):
    lens, seqs = make_stream(L, D, n_seq, 3, dtype)
    g = torch.Generator().manual_seed(5)
    ann_full = [torch.randint(-1, n_annot, (int(n), 2), generator=g) for n in lens]
    E = pilot_edges([torch.cat([s[l] for s in seqs[:120]]) for l in range(L)])
    cfg = AtlasConfig(K=K, Kr=Kr, rho_res=0.05, rho_sub=0.2, cap_frac=0.05, seed=11)
    halves = [list(range(0, n_seq, 2)), list(range(1, n_seq, 2))]  # interleaved, disjoint
    A = [Atlas(L, D, E, cfg, n_annot=n_annot) for _ in halves]
    shard_short = []
    if seeded:  # seeds from the GLOBAL stream (pilot = first 120 seqs, which the union contains)
        from atlas.reference import pilot_seeds
        ts, rk = pilot_seeds([torch.cat([s_[l] for s_ in seqs[:120]]) for l in range(L)], E, K, Kr, n_total=int(lens.sum()))
        for at in A:
            at.seed(ts, rk)
    for at, h in zip(A, halves):
        feed(at, seqs, lens, ann_full, h, D, L, dtype, g)
        shard_short.append(at.verify_seeds()["res_seed_limited_bins"])
    A[0].merge_from(A[1])
    got = A[0].result()
    if seeded:
        print(f"  info: seeded bins {got['res_seeded_bins']}; short per shard before merge {shard_short} "
              f"(expected: each shard holds half the data); short after merge {got['res_seed_limited_bins']}")
    Hall = [torch.cat([s[l] for s in seqs]) for l in range(L)]
    toks = torch.cat([pack_tok(torch.full((int(n),), j), torch.arange(int(n))) for j, n in enumerate(lens)])
    annf = torch.zeros(toks.numel(), n_annot)
    Aa = torch.cat(ann_full)
    r, c = torch.nonzero(Aa >= 0, as_tuple=True)
    annf[r, Aa[r, c]] = 1
    ref = reference(Hall, toks, E, K, Kr, cfg.rho_sub, cfg.seed, ann=annf, levels=cfg.enrich_levels)
    I = torch.iinfo(torch.int64).max
    okb = ~got["res_seed_limited"].unsqueeze(-1).expand_as(got["res_tok"])
    checks = {
        "top-K values": torch.equal(got["top_val"], ref["top_val"]),
        "reservoir keys (verified bins)": torch.equal(got["res_key"].where(got["res_tok"] >= 0, I)[okb], ref["res_key"][okb]),
        "reservoir tokens (verified bins)": torch.equal(got["res_tok"][okb], ref["res_tok"][okb]),
        "seed-limited bins after merge hold exactly the bin's keys under their seed": _limited_ok(got, ref),
        "bin masses": torch.equal(got["sub_bin"].double(), ref["sub_bin"]),
        "enrichment": torch.equal(got["enr"].double(), ref["enr"].double()),
        "n_sub": got["n_sub"] == ref["n_sub"],
    }
    tag = "seeded " if seeded else ""
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'} {tag}merged halves == offline: {k}")
    return all(checks.values())


def _limited_ok(got, ref):
    m = got["res_seed_limited"]
    if not bool(m.any()):
        return True
    MAX = torch.iinfo(torch.int64).max
    sk = got["res_seed_key"].unsqueeze(-1)
    want = torch.where(ref["res_key"] < sk, ref["res_key"], torch.full_like(ref["res_key"], MAX)).sort(-1).values
    have = got["res_key"].where(got["res_tok"] >= 0, MAX).sort(-1).values
    return torch.equal(have[m], want[m])


def test_resume(data, seeded=False):
    py = sys.executable
    with tempfile.TemporaryDirectory() as d:
        base = [py, "scripts/run_atlas.py", "--model", "8M", "--data", data, "--device", "cpu", "--dtype", "fp32",
                "--tokens-per-batch", "8192", "--pilot-tokens", "8192", "--K", "16", "--Kr", "8", "--max-seqs", "150",
                "--ckpt-every", "5"] + (["--seed-thresholds"] if seeded else [])
        full, part = Path(d) / "full", Path(d) / "part"
        # one thread: multi-threaded CPU SDPA is not bitwise reproducible on padded batches (the random
        # pilot batches are ~37% padding), and this test is about resume logic, not forward determinism
        env = dict(__import__("os").environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        run = lambda extra: subprocess.run(base + extra, check=True, capture_output=True, text=True, env=env)  # noqa: E731
        run(["--out", str(full)])
        r1 = run(["--out", str(part), "--stop-after", "7"])           # killed 2 batches after a checkpoint
        assert "stopping after 7" in r1.stdout, r1.stdout
        r2 = run(["--out", str(part), "--resume"])
        assert "resumed from ckpt_5.pt" in r2.stdout, r2.stdout
        A, B = torch.load(full / "atlas.pt", weights_only=False), torch.load(part / "atlas.pt", weights_only=False)
        keys = ["top_val", "top_tok", "res_tok", "res_val", "res_key", "sub_sum", "sub_sq", "sub_bin", "enr", "ann_tot"]
        ok = all(torch.equal(A[k], B[k]) if A[k].is_floating_point() is False or k != "res_val"
                 else torch.allclose(A[k], B[k], equal_nan=True, rtol=0, atol=0) for k in keys)
        ok &= A["n_seen"] == B["n_seen"]
        tag = "seeded " if seeded else ""
        print(f"  {'PASS' if ok else 'FAIL'} {tag}full run == stop-at-7 (ckpt at 5) + resume: {keys}, n_seen={A['n_seen']}")
        print(f"  info: seed verification full/resumed: {A['res_seed_limited_bins']}/{B['res_seed_limited_bins']} short of "
              f"{A['res_seeded_bins']} seeded; Gram rows full/resumed {A['gram_rows']}/{B['gram_rows']} (Gram is not checkpointed)")
        n_lines = sum(1 for _ in open(part / "batches.jsonl"))
        print(f"  info: batches.jsonl has {n_lines} lines for {sum(1 for _ in open(full / 'batches.jsonl'))} batches "
              f"(batches 5-6 logged twice: replayed after resume)")
        return ok


if __name__ == "__main__":
    ok = test_merge() & test_merge(seeded=True)
    data = sys.argv[sys.argv.index("--resume-data") + 1] if "--resume-data" in sys.argv else None
    if data:
        ok &= test_resume(data) & test_resume(data, seeded=True)
    sys.exit(0 if ok else 1)
