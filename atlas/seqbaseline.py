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


# --------------------------------------------------------------------------------------------- v1
# Baseline v1 (results/cases_round2_plan.md section 2; frozen 2026-11-05). Differences from v0:
#   - same level: the window preference r(w) = P(w | peak) / P(w | background) is fitted on the peak sample
#     itself, not on lower bins (a sequence-driven column can prefer different windows at different levels);
#   - a model that can represent residue conjunctions: an MLP over the one-hot +-5 window (11 x 21 -> 32 ReLU -> 1);
#   - fitting and evaluation separated by UniRef50 cluster: 5 folds; fold f's expectation uses background
#     tokens of fold f's clusters, reweighted by the model fitted without fold f;
#   - support limits: "not estimable" with < 40 peak tokens carrying the descriptor, or a fold whose
#     reweighting has an effective sample size < 100.
# Output wording: p <= 0.01 -> "unexplained by this fitted +-5 window model".

def _fold(cluster_idx: torch.Tensor, K: int) -> torch.Tensor:
    return ((cluster_idx.to(torch.int64) * 2654435761) % (2 ** 32)) % K


class WindowTablesV1(WindowTables):
    def __init__(self, T: Tables, W: int = 5, n_bg: int = 1_000_000, seed: int = 0, K: int = 5):
        super().__init__(T, W=W, n_bg=n_bg, seed=seed)
        self.K = K
        self.bg_fold = _fold(T.cluster[self.bg], K)


class _WinMLP(torch.nn.Module):
    def __init__(self, P: int, hidden: int = 32):
        super().__init__()
        self.emb = torch.nn.Parameter(torch.randn(P, PAD + 1, hidden) * 0.05)    # one-hot x W1, position by position
        self.b1 = torch.nn.Parameter(torch.zeros(hidden))
        self.w2 = torch.nn.Parameter(torch.randn(hidden) * 0.05)
        self.b2 = torch.nn.Parameter(torch.zeros(()))

    def forward(self, codes):                                                    # codes [n, P] in 0..20
        P = codes.shape[1]
        h = self.emb[torch.arange(P)[None, :], codes].sum(1) + self.b1
        return torch.relu(h) @ self.w2 + self.b2


def baseline_v1(WT: WindowTablesV1, res_tok_col: torch.Tensor, desc: str, B: int = 2000, seed: int = 0,
                n_neg: int = 5_000, hidden: int = 32, steps: int = 200, wd: float = 1e-2, lr: float = 1e-2,
                min_n: int = 40, min_ess: float = 100.0):
    T, K = WT.T, WT.K
    peak = dedup(T.rows(res_tok_col[5]), T.cluster)
    if peak.numel() == 0:
        return {"status": "not estimable", "reason": "empty peak"}
    x_all = T.desc[desc][peak]
    ev = x_all >= 0
    if int(ev.sum()) < min_n:
        return {"status": "not estimable", "reason": f"{int(ev.sum())} peak tokens with {desc} (< {min_n})"}
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    fp = _fold(T.cluster[peak], K)
    wp = WT.windows(peak)
    nk = int(T.desc[desc].max()) + 1
    x_bg = T.desc[desc][WT.bg]
    exp = torch.zeros(nk); per_fold = []
    for f in range(K):
        tr_pos = wp[fp != f]
        nb = torch.nonzero(WT.bg_fold != f).flatten()
        nb = nb[torch.randperm(nb.numel(), generator=g)[:n_neg]]
        X = torch.cat([tr_pos, WT.bg_win[nb]]); y = torch.cat([torch.ones(len(tr_pos)), torch.zeros(len(nb))])
        m = _WinMLP(X.shape[1], hidden)
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
        pw = torch.tensor(len(nb) / max(len(tr_pos), 1))
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m(X), y, pos_weight=pw)
            loss.backward(); opt.step()
        with torch.no_grad():
            ev_bg = torch.nonzero((WT.bg_fold == f) & (x_bg >= 0)).flatten()
            lw = torch.cat([m(WT.bg_win[ev_bg[i:i + 200_000]]) for i in range(0, ev_bg.numel(), 200_000)])
            w = torch.exp(lw - lw.max())
            pf = torch.zeros(nk).index_add_(0, x_bg[ev_bg], w); pf = pf / pf.sum()
            ess = float(w.sum() ** 2 / (w ** 2).sum())
        nf = int(((fp == f) & ev).sum())
        exp += nf * pf
        per_fold.append({"n_train_pos": int(len(tr_pos)), "n_eval": nf, "ess": round(ess, 1), "p": pf})
    ess_min = min(r["ess"] for r in per_fold)
    out = {"desc": desc, "n_peak": int(ev.sum()), "ess_min": ess_min, "n_train_pos_median": sorted(r["n_train_pos"] for r in per_fold)[K // 2]}
    if ess_min < min_ess:
        return dict(out, status="not estimable", reason=f"fold ESS {ess_min} (< {min_ess})")
    obs = torch.bincount(x_all[ev], minlength=nk).float()
    sim = torch.zeros(B, nk)
    for r in per_fold:
        if r["n_eval"]:
            d = torch.multinomial(r["p"].expand(B, nk), r["n_eval"], replacement=True, generator=g)
            sim.scatter_add_(1, d, torch.ones(B, r["n_eval"]))
    G = lambda c: (c * torch.log((c + 1e-9) / (exp + 1e-12))).sum(-1)  # noqa: E731
    p = float(((G(sim) >= G(obs)).sum() + 1) / (B + 1))
    return dict(out, status="ok", p_unexplained=p,
                verdict="unexplained by this fitted +-5 window model" if p <= 0.01 else "consistent with the fitted window model (sequence-only not excluded)",
                log2_obs_over_window=[round(float(v), 2) for v in torch.log2((obs + 0.5) / (exp + 0.5))])
