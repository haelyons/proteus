"""Local-sequence baseline v0 (declared 2026-11-05 in results/ruler_challenges_predictions.md; frozen).

A diagnostic reported next to ruler calls, not part of them: is a column's association with a structural
descriptor explained by the sequence window around the sampled residues?

  1. an independent-sites model of the +-W window (2W+1 positions x 21 codes: 20 residues + outside the
     protein), log-odds against background frequencies at each offset, with `lam` pseudo-counts per offset,
     fitted on the column's q99.9-q99.999 reservoir tokens (bins 3-4, one per UniRef50 cluster);
  2. a background sample of residues reweighted by exp(sum of log-odds) gives the descriptor distribution a
     sequence-only column with that window preference would show;
  3. the peak sample (bin 5, disjoint from the fit) is tested against it: Monte Carlo G, `B` draws.
A small p means structure beyond what the window predicts. Known limits: independent sites cannot represent a
conjunction of residues exactly; a wider window absorbs more genuine structure (W = 5 is a choice).
"""
from __future__ import annotations

import torch

from atlas.ruler import Tables, dedup

PAD = 20


class WindowTables:
    def __init__(self, T: Tables, W: int = 5, n_bg: int = 200_000, seed: int = 0):
        self.T, self.W = T, W
        n = T.n
        self.pos = torch.empty(n, dtype=torch.long)
        self.len = torch.empty(n, dtype=torch.long)
        for q, base in T.row_of.items():
            L = T.len_of[q]
            self.pos[base:base + L] = torch.arange(L)
            self.len[base:base + L] = L
        g = torch.Generator().manual_seed(seed)
        ok = torch.nonzero(T.aa >= 0).flatten()
        self.bg = ok[torch.randperm(ok.numel(), generator=g)[:n_bg]]
        self.bg_win = self.windows(self.bg)
        self.f_bg = torch.stack([torch.bincount(self.bg_win[:, k], minlength=PAD + 1).float() for k in range(2 * W + 1)])
        self.f_bg = (self.f_bg + 1) / (self.f_bg + 1).sum(1, keepdim=True)          # [2W+1, 21]

    def windows(self, rows: torch.Tensor) -> torch.Tensor:
        off = torch.arange(-self.W, self.W + 1)
        p = self.pos[rows][:, None] + off[None, :]
        inside = (p >= 0) & (p < self.len[rows][:, None])
        r = (rows[:, None] + off[None, :]).clamp(0, self.T.n - 1)
        a = self.T.aa[r].clone()
        a[~inside] = PAD
        a[a < 0] = PAD
        return a


def baseline_test(WT: WindowTables, res_tok_col: torch.Tensor, desc: str, B: int = 2000, lam: float = 20.0, seed: int = 0):
    """res_tok_col [n_bins, Kr] for one column (positive pole). Returns the test of the peak's `desc` counts
    against the window-reweighted background, or None when a sample is too small."""
    T = WT.T
    fit = dedup(T.rows(torch.cat([res_tok_col[3], res_tok_col[4]])), T.cluster)
    peak = dedup(T.rows(res_tok_col[5]), T.cluster)
    if fit.numel() < 10 or peak.numel() < 10:
        return None
    K = 2 * WT.W + 1
    wf = WT.windows(fit)
    c = torch.stack([torch.bincount(wf[:, k], minlength=PAD + 1).float() for k in range(K)])
    lo = torch.log((c + lam * WT.f_bg) / (fit.numel() + lam)) - torch.log(WT.f_bg)  # [K, 21]
    lw = lo[torch.arange(K)[None, :], WT.bg_win].sum(1)                            # [n_bg]
    w = torch.exp(lw - lw.max())
    x_bg = T.desc[desc][WT.bg]
    m = x_bg >= 0
    nk = int(T.desc[desc].max()) + 1
    pb = torch.zeros(nk).index_add_(0, x_bg[m], w[m])
    pb = pb / pb.sum()
    ess = float(w[m].sum() ** 2 / (w[m] ** 2).sum())
    x = T.desc[desc][peak]
    x = x[x >= 0]
    n = int(x.numel())
    if n < 10:
        return None
    obs = torch.bincount(x, minlength=nk).float()
    exp = pb * n
    gen = torch.Generator().manual_seed(seed)
    sim = torch.multinomial(pb.expand(B, nk), n, replacement=True, generator=gen)
    simc = torch.zeros(B, nk).scatter_add_(1, sim, torch.ones(B, n))
    G = lambda cnt: (cnt * torch.log((cnt + 1e-9) / (exp + 1e-12))).sum(-1)  # noqa: E731
    p = float(((G(simc) >= G(obs)).sum() + 1) / (B + 1))
    return {"desc": desc, "n_fit": int(fit.numel()), "n_peak": n, "p_beyond_window": p, "ess": round(ess, 1),
            "log2_obs_over_window": [round(float(v), 2) for v in torch.log2((obs + 0.5) / (exp + 0.5))]}
