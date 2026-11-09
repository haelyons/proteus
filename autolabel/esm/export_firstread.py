"""Export compact first-read bundles from a proteus atlas, for the autolabel ESM notebook.

Runs where the atlas lives (a Lambda box with esm-atlas-results mounted); CPU only. Self-contained (no proteus
import): token ids are seq_idx * 1024 + residue position (proteus atlas/core.py SEQ_STRIDE).

  python -I export_firstread.py bundle --model 8M --atlas A.pt [--control C.pt] --data swissprot.pt --desc desc.pt \
      --out DIR [--res-per-bin 16] [--peers 8] [--readers 8] [--hf-model NAME_OR_DIR]
  python -I export_firstread.py sequences --data swissprot.pt --out DIR
  python -I export_firstread.py selftest

bundle -> DIR/firstread_<model>.pt with, per (layer, pole, unit): top-K values and tokens; reservoir samples
(first R per bin = R smallest hash keys = a uniform sample of the bin); bin edges and masses; mean and sd on the
hashed subsample; co-activation peers (Pearson r from the Gram); input-weight peers (cosine of W_in rows); direct
logit effect of W_out on the 20 amino-acid tokens (tied embedding, LayerNorm and lm_head.dense ignored); readers in
the next two layers (cosine of W_in rows of layer l+1/l+2 with this unit's W_out column, residual path only);
annotation enrichment (top 10 per level); AlphaFold DB descriptors at every exported token; base rates.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SEQ_STRIDE = 1024
AA = "ACDEFGHIKLMNPQRSTVWY"
MODELS = {"8M": "facebook/esm2_t6_8M_UR50D", "35M": "facebook/esm2_t12_35M_UR50D",
          "150M": "facebook/esm2_t30_150M_UR50D", "650M": "facebook/esm2_t33_650M_UR50D",
          "3B": "facebook/esm2_t36_3B_UR50D"}


def log(*a):
    print(f"[export {time.strftime('%H:%M:%S')}]", *a, flush=True)


# ------------------------------------------------------------------ pure pieces (selftested)
def pearson_from_gram(G: torch.Tensor, s: torch.Tensor, sq: torch.Tensor, n: float) -> torch.Tensor:
    """Pearson r [D, D] from Gram sum h_i h_j, sums and sums of squares over n rows."""
    G, s, sq = G.double(), s.double(), sq.double()
    mu = s / n
    cov = G / n - mu[:, None] * mu[None, :]
    sd = torch.sqrt(torch.clamp(sq / n - mu * mu, min=1e-30))
    return (cov / (sd[:, None] * sd[None, :])).float()


def top_offdiag(M: torch.Tensor, k: int):
    """Per row, the k largest entries excluding the diagonal: (idx int32, val fp16)."""
    M = M.clone()
    M.fill_diagonal_(-float("inf"))
    v, i = torch.topk(M, k, dim=1)
    return i.to(torch.int32), v.to(torch.float16)


def top_abs(M: torch.Tensor, k: int):
    """Per column j of M [rows, cols], the k rows with the largest |M| (signed values kept)."""
    v, i = torch.topk(M.abs().t(), k, dim=1)
    return i.to(torch.int32), torch.gather(M.t(), 1, i).to(torch.float16)


def residue_index(tok: torch.Tensor, desc_off: torch.Tensor, desc_len: torch.Tensor) -> torch.Tensor:
    """Packed token ids -> index into concatenated descriptor arrays, -1 where unavailable."""
    t = tok.long()
    ok = t >= 0
    q = torch.where(ok, t // SEQ_STRIDE, torch.zeros_like(t))
    p = torch.where(ok, t % SEQ_STRIDE, torch.zeros_like(t))
    off, ln = desc_off[q], desc_len[q]
    good = ok & (off >= 0) & (p < ln)
    return torch.where(good, off + p, torch.full_like(t, -1))


def lookup(desc: dict, idx: torch.Tensor) -> dict:
    """Descriptor values at residue indices (-1 -> missing): ss3 int8 (-1 missing), burial/contact/plddt uint8."""
    safe = idx.clamp(min=0)
    miss = idx < 0
    out = {}
    out["ss3"] = torch.where(miss, torch.full_like(idx, -1), desc["ss3"][safe].long()).to(torch.int8)
    for k in ("burial", "contact", "plddt"):
        v = desc[k][safe].float().round().clamp(0, 255)
        out[k] = torch.where(miss, torch.zeros_like(v), v).to(torch.uint8)
    return out


def enrichment(enr: torch.Tensor, bin_mass: torch.Tensor, ann_tot: torch.Tensor, n_sub: float, levels, k: int = 10, min_count: int = 3):
    """enr [L, 2, nl, D, A] counts of annotated subsample tokens with s >= E[level]; bin_mass [L, 2, D, B].
    Returns top-k annotations by enrichment ratio per (l, p, level, unit) with count >= min_count:
    ids int16 (-1 pad), counts int32, ratio fp16, and above [L, 2, nl, D] = subsample tokens over each level."""
    above = torch.stack([bin_mass[..., lev:].sum(-1) for lev in levels], 2).double()  # [L, 2, nl, D]
    base = (ann_tot.double() + 0.5) / (n_sub + 1.0)                                    # [A]
    ratio = ((enr.double() + 0.5) / (above[..., None] + 1.0)) / base
    ratio = torch.where(enr >= min_count, ratio, torch.zeros_like(ratio))
    v, i = torch.topk(ratio, min(k, ratio.shape[-1]), dim=-1)
    cnt = torch.gather(enr, -1, i)
    i = torch.where(v > 0, i, torch.full_like(i, -1))
    return i.to(torch.int16), cnt.to(torch.int32), v.to(torch.float16), above.to(torch.int64)


# ------------------------------------------------------------------ bundle
def load_desc(path, n_seq):
    d = torch.load(path, map_location="cpu", weights_only=False)
    off = torch.full((n_seq,), -1, dtype=torch.long)
    ln = torch.zeros(n_seq, dtype=torch.long)
    si = d["seq_idx"].long()
    o = d["offset"].long()
    off[si] = o[:-1]
    ln[si] = o[1:] - o[:-1]
    base = {}
    ss3, pl = d["ss3"].long(), d["plddt"].float()
    conf = pl >= 70
    base["ss3_frac_all"] = [float((ss3 == c).float().mean()) for c in (0, 1, 2)]
    base["ss3_frac_conf"] = [float((ss3[conf] == c).float().mean()) for c in (0, 1, 2)]
    base["burial_mean"] = float(d["burial"].float().mean())
    base["burial_q"] = [float(x) for x in torch.quantile(d["burial"].float()[::97], torch.tensor([0.1, 0.5, 0.9]))]
    base["contact_mean"] = float(d["contact"].float().mean())
    base["plddt_mean"] = float(pl.mean())
    base["plddt_lt50"] = float((pl < 50).float().mean())
    base["n_residues"] = int(ss3.numel())
    return {k: d[k] for k in ("ss3", "burial", "contact", "plddt")}, off, ln, base


def bundle(a):
    torch.set_grad_enabled(False)
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log("loading atlas", a.atlas, f"{os.path.getsize(a.atlas) / 1e9:.2f} GB")
    at = torch.load(a.atlas, map_location="cpu", weights_only=False)
    L, P, D, K = at["top_tok"].shape
    B = at["edges"].shape[-1]
    R = min(a.res_per_bin, at["res_tok"].shape[-1])
    n_sub = float(at["n_sub"])
    cfg = at.get("cfg", {}) if isinstance(at.get("cfg"), dict) else {}
    levels = list(cfg.get("enrich_levels", (2, 3, 4)))
    log(f"L={L} D={D} K={K} B={B} Kr={at['res_tok'].shape[-1]} R={R} n_seen={at['n_seen']} n_sub={int(n_sub)}")
    b = {"meta": {"model": a.model, "atlas": a.atlas, "atlas_bytes": os.path.getsize(a.atlas), "L": L, "D": D, "K": K, "B": B, "R": R,
                  "n_seen": int(at["n_seen"]), "n_sub": int(n_sub), "cfg": cfg, "enrich_levels": levels,
                  "bin_quantiles": [0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999][:B], "seq_stride": SEQ_STRIDE,
                  "notes": "pre-GELU FFN activation (intermediate.dense output); pole 1 values are negative (pole-oriented -h)"}}
    b["top_val"] = at["top_val"].to(torch.float16)
    b["top_tok"] = at["top_tok"].to(torch.int32)
    b["edges"] = at["edges"].float()
    b["bin_mass"] = at["sub_bin"].to(torch.int32)
    b["res_tok"] = at["res_tok"][..., :R].to(torch.int32).contiguous()
    b["res_val"] = at["res_val"][..., :R].to(torch.float16).contiguous()
    mu = at["sub_sum"].double() / n_sub
    b["mean"] = mu.float()
    b["sd"] = torch.sqrt(torch.clamp(at["sub_sq"].double() / n_sub - mu * mu, min=0)).float()
    if "gram" in at:
        pi, pv = [], []
        for l in range(L):
            r = pearson_from_gram(at["gram"][l], at["sub_sum"][l], at["sub_sq"][l], n_sub)
            i, v = top_offdiag(r, a.peers); pi.append(i); pv.append(v)
        b["peer_coact_idx"], b["peer_coact_r"] = torch.stack(pi), torch.stack(pv)
        log("co-activation peers done")
    if "enr" in at and "ann_tot" in at:
        ids, cnt, ratio, above = enrichment(at["enr"].long(), at["sub_bin"].long(), at["ann_tot"].long(), n_sub, levels)
        b["enr_ids"], b["enr_count"], b["enr_ratio"], b["enr_above"] = ids, cnt, ratio, above
        b["ann_tot"] = at["ann_tot"].long()
        log("enrichment done")
    del at
    # weights
    from transformers import EsmModel
    name = a.hf_model or MODELS[a.model]
    log("loading weights", name)
    model = EsmModel.from_pretrained(name, add_pooling_layer=False, torch_dtype=torch.float32)
    lay = model.encoder.layer
    assert len(lay) == L and lay[0].intermediate.dense.weight.shape[0] == D, "atlas and model shapes differ"
    from transformers import EsmTokenizer
    tok = EsmTokenizer.from_pretrained(name)
    aa_ids = torch.tensor([tok.convert_tokens_to_ids(c) for c in AA])
    E = model.embeddings.word_embeddings.weight[aa_ids].float()                      # [20, d]
    win = [lay[l].intermediate.dense.weight.float() for l in range(L)]               # [D, d]
    wout = [lay[l].output.dense.weight.float() for l in range(L)]                    # [d, D]
    nin = [torch.nn.functional.normalize(w, dim=1) for w in win]
    nout = [torch.nn.functional.normalize(w, dim=0) for w in wout]
    pi, pv, lg = [], [], []
    for l in range(L):
        i, v = top_offdiag(nin[l] @ nin[l].t(), a.peers); pi.append(i); pv.append(v)
        lg.append((E @ wout[l]).t().to(torch.float16))                              # [D, 20]
    b["peer_input_idx"], b["peer_input_cos"] = torch.stack(pi), torch.stack(pv)
    b["logit_aa"] = torch.stack(lg)
    ri = torch.full((L, 2, D, a.readers), -1, dtype=torch.int32)
    rv = torch.zeros((L, 2, D, a.readers), dtype=torch.float16)
    for l in range(L):
        for h in (1, 2):
            if l + h < L:
                i, v = top_abs(nin[l + h] @ nout[l], a.readers)                     # rows: readers in l+h
                ri[l, h - 1], rv[l, h - 1] = i, v
    b["reader_idx"], b["reader_cos"] = ri, rv
    b["meta"]["aa_order"] = AA
    b["meta"]["wiring_notes"] = "input/output cosines and readers ignore LayerNorm and attention; logit_aa = tied embedding @ W_out column"
    log("wiring done")
    del model, win, wout, nin, nout
    # descriptors at exported tokens
    seqd = torch.load(a.data, map_location="cpu", weights_only=False)
    n_seq = len(seqd["seq"])
    del seqd
    desc, off, ln, base = load_desc(a.desc, n_seq)
    b["base"] = base
    for name_, t in (("top", b["top_tok"]), ("res", b["res_tok"])):
        idx = residue_index(t.reshape(-1), off, ln)
        vals = lookup(desc, idx)
        for k, v in vals.items():
            b[f"{name_}_{k}"] = v.view(t.shape)
        b["meta"][f"{name_}_desc_coverage"] = float((idx >= 0).float().mean())
    log("descriptors done; coverage top", round(b["meta"]["top_desc_coverage"], 3), "res", round(b["meta"]["res_desc_coverage"], 3))
    if a.control:
        log("loading control", a.control)
        c = torch.load(a.control, map_location="cpu", weights_only=False)
        b["ctrl_top_val"] = c["top_val"].to(torch.float16)
        b["ctrl_top_tok"] = c["top_tok"].to(torch.int32)
        b["ctrl_edges"] = c["edges"].float()
        b["meta"]["control"] = a.control
        del c
    path = out / f"firstread_{a.model}.pt"
    torch.save(b, path)
    log("saved", path, f"{os.path.getsize(path) / 1e9:.3f} GB in {time.time() - t0:.0f} s")


def sequences(a):
    d = torch.load(a.data, map_location="cpu", weights_only=False)
    seqs = d["seq"]
    lens = np.array([len(s) for s in seqs], dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(lens)])
    blob = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8)
    iv = d["iv"].numpy() if torch.is_tensor(d["iv"]) else np.asarray(d["iv"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    path = out / "sequences.npz"
    np.savez_compressed(path, blob=blob, off=off, acc=np.array(d["acc"]), iv=iv, vocab=np.array([str(v) for v in d["vocab"]]))
    log("saved", path, f"{os.path.getsize(path) / 1e6:.1f} MB; {len(seqs)} sequences; iv {iv.shape}; vocab {len(d['vocab'])}")


# ------------------------------------------------------------------ selftest
def selftest():
    torch.manual_seed(0)
    H = torch.randn(500, 6)
    H[:, 1] = H[:, 0] * 0.9 + 0.1 * torch.randn(500)
    r = pearson_from_gram(H.t() @ H, H.sum(0), (H * H).sum(0), 500)
    ref = torch.corrcoef(H.t())
    assert torch.allclose(r, ref, atol=1e-5), (r, ref)
    i, v = top_offdiag(r, 2)
    assert int(i[0, 0]) == 1 and int(i[1, 0]) == 0 and all(int(i[k, 0]) != k for k in range(6))
    M = torch.tensor([[0.1, -0.9], [0.5, 0.2], [-0.3, 0.8]])
    ti, tv = top_abs(M, 2)
    assert ti[0].tolist() == [1, 2] and abs(float(tv[0, 0]) - 0.5) < 1e-3
    assert ti[1].tolist() == [0, 2] and abs(float(tv[1, 0]) + 0.9) < 1e-3
    off = torch.tensor([0, -1, 10]); ln = torch.tensor([10, 0, 5])
    tok = torch.tensor([3, 1 * SEQ_STRIDE + 2, 2 * SEQ_STRIDE + 4, 2 * SEQ_STRIDE + 5, -1])
    assert residue_index(tok, off, ln).tolist() == [3, -1, 14, -1, -1]
    desc = {"ss3": torch.arange(15) % 3, "burial": torch.arange(15).float() * 30, "contact": torch.ones(15), "plddt": torch.full((15,), 91.6)}
    lk = lookup(desc, torch.tensor([3, -1, 14]))
    assert lk["ss3"].tolist() == [0, -1, 2] and lk["burial"].tolist() == [90, 0, 255] and lk["plddt"].tolist() == [92, 0, 92]
    enr = torch.zeros(1, 2, 1, 2, 3, dtype=torch.long); enr[0, 0, 0, 0] = torch.tensor([10, 1, 0])
    bm = torch.zeros(1, 2, 2, 4, dtype=torch.long); bm[0, 0, 0] = torch.tensor([80, 10, 6, 4])
    ids, cnt, ratio, above = enrichment(enr, bm, torch.tensor([100, 50, 10]), 1000.0, [2], k=2)
    assert int(above[0, 0, 0, 0]) == 10 and int(ids[0, 0, 0, 0, 0]) == 0 and int(cnt[0, 0, 0, 0, 0]) == 10
    assert int(ids[0, 0, 0, 0, 1]) == -1, "count below min_count must not rank"
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("bundle")
    p.add_argument("--model", required=True); p.add_argument("--atlas", required=True); p.add_argument("--control")
    p.add_argument("--data", required=True); p.add_argument("--desc", required=True); p.add_argument("--out", required=True)
    p.add_argument("--res-per-bin", type=int, default=16); p.add_argument("--peers", type=int, default=8)
    p.add_argument("--readers", type=int, default=8); p.add_argument("--hf-model")
    s = sub.add_parser("sequences"); s.add_argument("--data", required=True); s.add_argument("--out", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"bundle": bundle, "sequences": sequences, "selftest": lambda a: selftest()}[a.cmd](a)


if __name__ == "__main__":
    main()
