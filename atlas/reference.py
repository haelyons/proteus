"""Offline ground truth for every streamed quantity (materialises everything; tests only)."""
from __future__ import annotations

import torch

from .core import bin_of, token_keys


def reference(Hs: list[torch.Tensor], toks: torch.Tensor, edges: torch.Tensor, K: int, Kr: int,
              rho_sub: float, seed: int = 0, ann: torch.Tensor | None = None, levels=(2, 3, 4)):
    """Hs: per layer [N, D] valid-token activations (same dtype as streamed). toks [N].
    Returns dict with the same keys/shapes as Atlas.result() (subset)."""
    L = len(Hs)
    N, D = Hs[0].shape
    B = edges.shape[-1]
    key = token_keys(toks, seed)
    sub = key < int(rho_sub * 2**32)
    I64MAX = torch.iinfo(torch.int64).max
    out = {k: [] for k in ["top_val", "top_tok", "res_key", "res_tok", "res_val", "sub_sum", "sub_sq", "sub_bin", "gram", "enr"]}
    for l in range(L):
        H = Hs[l].float()
        tv, tt, rk, rt, rv, sb, en = [], [], [], [], [], [], []
        for p, sgn in ((0, 1.0), (1, -1.0)):
            S = H * sgn
            v, i = torch.topk(S, K, dim=0)
            tv.append((v * sgn).T); tt.append(toks[i].T)
            b = bin_of(S, edges[l, p])
            kb, tb, vb = [], [], []
            for bb in range(B):
                km = torch.where(b == bb, key[:, None], torch.full_like(b, I64MAX))
                kv, ki = torch.topk(km, Kr, dim=0, largest=False)
                empty = kv == I64MAX
                kb.append(kv.T); tb.append(torch.where(empty, -1, toks[ki]).T)
                vb.append(torch.where(empty, torch.nan, H.gather(0, ki)).T)
            rk.append(torch.stack(kb, 1)); rt.append(torch.stack(tb, 1)); rv.append(torch.stack(vb, 1))
            bs = b[sub]
            sb.append(torch.stack([(bs == bb).sum(0) for bb in range(B)], 1).double())
            if ann is not None:
                Ss = S[sub]
                en.append(torch.stack([(Ss >= edges[l, p, :, lev]).float().T @ ann[sub] for lev in levels]))
        out["top_val"].append(torch.stack(tv)); out["top_tok"].append(torch.stack(tt))
        out["res_key"].append(torch.stack(rk)); out["res_tok"].append(torch.stack(rt)); out["res_val"].append(torch.stack(rv))
        Hsub = H[sub]
        out["sub_sum"].append(Hsub.double().sum(0)); out["sub_sq"].append((Hsub.double() ** 2).sum(0))
        out["sub_bin"].append(torch.stack(sb))
        out["gram"].append(Hsub.double().T @ Hsub.double())
        if ann is not None:
            out["enr"].append(torch.stack(en))
    res = {k: torch.stack(v) for k, v in out.items() if v}
    res["n_sub"] = int(sub.sum())
    return res


def pilot_edges(Hs: list[torch.Tensor], probs=(0.5, 0.9, 0.99, 0.999, 0.9999), chunk: int = 512) -> torch.Tensor:
    """Per (layer, pole, neuron) lower bin edges from a pilot sample: E[l, p, d, b] is the
    probs[b]-quantile of s = +h (p=0) / -h (p=1). Returns [L, 2, D, B] fp32."""
    out = []
    for H in Hs:
        n, D = H.shape
        idx = torch.tensor([min(n - 1, int(round(q * (n - 1)))) for q in probs], device=H.device)
        per = []
        for sgn in (1.0, -1.0):
            cols = []
            for c0 in range(0, D, chunk):  # column chunks: a full-width sort needs several x the input in scratch
                s, _ = torch.sort(H[:, c0:c0 + chunk].float() * sgn, dim=0)
                cols.append(s.index_select(0, idx).T)
                del s
            per.append(torch.cat(cols))  # [D, B]
        out.append(torch.stack(per))
    E = torch.stack(out)
    # enforce strictly increasing edges per (l, p, d) so bins are well defined under ties
    for b in range(1, E.shape[-1]):
        E[..., b] = torch.maximum(E[..., b], torch.nextafter(E[..., b - 1], torch.full_like(E[..., b - 1], float("inf"))))
    return E
