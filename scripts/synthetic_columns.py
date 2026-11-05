"""Synthetic columns with known jobs, streamed through the same Atlas code as model units (ruler validation).

Per residue (proteins with an AlphaFold model in the descriptor subset), every column is
    z = alpha * signal + noise,  noise ~ N(0, 1) independent per column,
over an SNR ladder alpha. The atlas is rank-based (quantile bins, top-K), so the marginal shape does not
matter; what matters is how the signal moves tokens into the tail. Column types:
  helix        one structural job: 1[state == helix]                         (pLDDT >= 70 for helix/strand/loop)
  buried       one structural job, ordinal: burial count, standardised        (pLDDT >= 70, else 0)
  trp          composition job: 1[residue == W]
  lys_helix    context job: 1[residue == K and state == helix]
  two_job      planted mixture: helix in UniRef50 clusters of group A, strand in group B
  ordered      one meaning, two contexts: 1[state in {helix, strand}]
  narrow       peak = helix N-cap S/T/D/N (first residue of a helix run), weaker level = any S/T (alpha/2)
  hydro_win    sequence-only: mean Kyte-Doolittle hydropathy of the +-3 window, standardised
  noise        alpha ignored: pure noise
Output: atlas result (top-K, reservoirs, bins, sums) for L = 1 layer of C columns + column table.

  python scripts/synthetic_columns.py --data swissprot.pt --desc-dir desc_subset/ --uniref lists/..tsv.gz --out synth/
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
import time
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.core import Atlas, AtlasConfig, pack_tok  # noqa: E402
from atlas.reference import pilot_edges  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
KD = dict(zip(AA, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
TYPES = ["helix", "buried", "trp", "lys_helix", "two_job", "ordered", "narrow", "hydro_win", "noise"]
ALPHAS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]


def load_subset(desc_dir, data, uniref, heldout=False):
    parts = sorted(glob.glob(str(Path(desc_dir) / "part_*.pt")))
    seqs = data["seq"]
    cl = {}
    with gzip.open(uniref, "rt") as f:
        next(f)
        for ln in f:
            a, c = ln.rstrip("\n").split("\t"); cl[a] = c
    prot = []
    for p in parts:
        d = torch.load(p, weights_only=False)
        off = d["offset"]
        for j, acc in enumerate(d["acc"]):
            if d["heldout"][j] != heldout:
                continue
            s, e = int(off[j]), int(off[j + 1])
            prot.append({"acc": acc, "seq_idx": int(d["seq_idx"][j]), "seq": seqs[int(d["seq_idx"][j])][: e - s],
                         "ss3": d["ss3"][s:e], "plddt": d["plddt"][s:e].float(), "burial": d["burial"][s:e].float(),
                         "phi": d["phi"][s:e].float(), "psi": d["psi"][s:e].float(), "cluster": cl.get(acc) or acc})
    return prot


def state5(p):
    """0 loop, 1 helix, 2 strand (pLDDT >= 70); 3 disordered (< 50); 4 uncertain (50-70)."""
    st = p["ss3"].long().clone()
    st[p["plddt"] < 70] = 4
    st[p["plddt"] < 50] = 3
    return st


def signals(p, burial_mu, burial_sd):
    seq = p["seq"]; n = len(seq); st = state5(p)
    aa = torch.tensor([AA.find(c) for c in seq])
    helix = (st == 1).float(); strand = (st == 2).float()
    grpA = (zlib.crc32(p["cluster"].encode()) % 2) == 0
    starts = torch.zeros(n)
    hb = (st == 1)
    starts[0] = float(hb[0]); starts[1:] = (hb[1:] & ~hb[:-1]).float()
    ncap = starts * torch.tensor([float(c in "STDN") for c in seq])
    st_res = torch.tensor([float(c in "ST") for c in seq])
    bur = torch.where(p["plddt"] >= 70, (p["burial"] - burial_mu) / burial_sd, torch.zeros(n))
    kd = torch.tensor([KD.get(c, 0.0) for c in seq])
    win = torch.nn.functional.avg_pool1d(kd[None, None], 7, 1, 3, count_include_pad=False)[0, 0]
    return {"helix": helix, "buried": bur, "trp": (aa == AA.index("W")).float(),
            "lys_helix": ((aa == AA.index("K")).float() * helix), "two_job": helix if grpA else strand,
            "ordered": helix + strand, "narrow": ncap, "narrow_weak": st_res, "hydro_win": (win + 0.5) / 1.6,
            "noise": torch.zeros(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--desc-dir", required=True)
    ap.add_argument("--uniref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens-per-batch", type=int, default=32768)
    ap.add_argument("--pilot-tokens", type=int, default=1_000_000)
    ap.add_argument("--Kr", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replicates", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = torch.load(a.data, weights_only=False)
    prot = load_subset(a.desc_dir, data, a.uniref)
    g = torch.Generator().manual_seed(a.seed)
    prot = [prot[i] for i in torch.randperm(len(prot), generator=g).tolist()]
    allb = torch.cat([p["burial"][p["plddt"] >= 70] for p in prot])
    bmu, bsd = float(allb.mean()), float(allb.std())
    cols = [(t, al, r) for t in TYPES for al in ALPHAS for r in range(a.replicates)]  # replicates: independent noise
    C = len(cols)
    (out / "columns.json").write_text(json.dumps([{"col": i, "type": t, "alpha": al, "rep": r} for i, (t, al, r) in enumerate(cols)], indent=0))

    def column_block(p, gen):
        sg = signals(p, bmu, bsd)
        n = len(p["seq"])
        Z = torch.randn(n, C, generator=gen)
        for i, (t, al, _) in enumerate(cols):
            if t == "noise" or al == 0:
                continue
            Z[:, i] += al * sg[t] + (0.5 * al * sg["narrow_weak"] if t == "narrow" else 0.0)
        return Z

    gen = torch.Generator().manual_seed(a.seed + 1)
    blocks, toks, n_tok = [], [], 0
    pilot = []
    for p in prot:  # pilot: the first tokens of the (shuffled) stream, then streamed again like the model runs
        if n_tok >= a.pilot_tokens:
            break
        Z = column_block(p, torch.Generator().manual_seed(p["seq_idx"]))
        pilot.append(Z); n_tok += Z.shape[0]
    E = pilot_edges([torch.cat(pilot)], probs=(0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999))
    at = Atlas(1, C, E, AtlasConfig(K=64, Kr=a.Kr, backend="torch", seed=a.seed, gram=False, cap_frac=0.02), n_annot=0)
    t0 = time.time(); nb = 0

    def flush(blocks, toks):
        H = torch.cat(blocks); tk = torch.cat(toks); n = H.shape[0]
        ctx = at.make_ctx(n, torch.arange(n), tk, None)
        at.begin_batch(ctx); at.observe(0, H)
        redo = at.end_batch()
        if redo:
            at.begin_batch(ctx, redo_layers=redo); at.observe(0, H); at.end_batch()
    cur = 0
    for p in prot:
        Z = column_block(p, torch.Generator().manual_seed(p["seq_idx"]))   # same noise as in the pilot (fixed per protein)
        blocks.append(Z); toks.append(pack_tok(torch.full((Z.shape[0],), p["seq_idx"]), torch.arange(Z.shape[0])))
        cur += Z.shape[0]
        if cur >= a.tokens_per_batch:
            flush(blocks, toks); blocks, toks, cur = [], [], 0; nb += 1
    if blocks:
        flush(blocks, toks); nb += 1
    res = at.result()
    res.update({"columns": [{"col": i, "type": t, "alpha": al, "rep": r} for i, (t, al, r) in enumerate(cols)], "n_proteins": len(prot)})
    torch.save(res, out / "synthetic_atlas.pt")
    print(json.dumps({"proteins": len(prot), "tokens": at.n_seen, "columns": C, "batches": nb, "seconds": round(time.time() - t0, 1),
                      "stats": at.stats}))


if __name__ == "__main__":
    main()
