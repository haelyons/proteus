"""Streaming per-neuron atlas: exact top-K, exact quantile-stratified bottom-k
reservoirs, and hashed-subsample statistics (moments, enrichment, Gram).

Everything is exact w.r.t. a stated definition:

* top-K   : for every (layer, pole, neuron) the K largest pole-oriented values
            s = +h (pole 0) / -h (pole 1) over all valid tokens, ties broken by the
            smaller token id (a total order: top_key(s, tok)). Exact and deterministic:
            the running K-th key only rises, so `s >= tau` admits every token that can
            still enter, whatever the arrival order.
* reservoir: for every (layer, pole, neuron, bin) the K_r tokens with the smallest
            32-bit key u(tok) among tokens whose s falls in that bin ("bottom-k").
            u is a bijective hash of the token id, so samples are uniform within
            the bin, reproducible, and collision-free. Same monotone argument.
* subsample: tokens with u < rho_sub * 2^32 form a fixed uniform hashed subsample;
            moments, bin masses, annotation contingency counts and the Gram matrix
            are computed exactly on it (estimates of full-data quantities).

Bins: per (layer, pole, neuron) lower edges E[..., 0..B-1], increasing; bin(s) =
#{b : s >= E_b} - 1, -1 meaning "not in this pole". Edges come from a pilot pass.

Per batch, the forward hooks do only full-pass work that needs no host sync
(one compare-and-compact over H plus row gathers of the hashed subsample).
Merges run once after the forward, stacked over all layers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

NEG_INF = float("-inf")
I64_MIN = torch.iinfo(torch.int64).min
M32 = 0xFFFFFFFF
SEQ_STRIDE = 1024  # tok id = seq_idx * 1024 + residue position (0-based, < 1022)


# --------------------------------------------------------------------------- hashing
def _mul32(x: torch.Tensor, c: int) -> torch.Tensor:
    """(x * c) mod 2^32 for 0 <= x < 2^32, exact in int64 (no overflow)."""
    lo, hi = c & 0xFFFF, c >> 16
    return (x * lo + (((x * hi) & 0xFFFF) << 16)) & M32


def token_keys(tok: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Bijective 32-bit mix (Wellons' lowbias32) of token ids < 2^32 -> int64 in [0, 2^32).
    Bijective => distinct tokens never share a key (no ties anywhere)."""
    x = (tok.to(torch.int64) ^ (seed & M32)) & M32
    x = x ^ (x >> 16)
    x = _mul32(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = _mul32(x, 0x846CA68B)
    x = x ^ (x >> 16)
    return x


def pack_tok(seq_idx: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    return seq_idx.to(torch.int64) * SEQ_STRIDE + pos.to(torch.int64)


def unpack_tok(tok: torch.Tensor):
    return tok // SEQ_STRIDE, tok % SEQ_STRIDE


def top_key(s: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
    """int64 key ordering tokens by (s descending, token id ascending): a strict total order,
    so the top-K is unique. s float (-0.0 is canonicalised to +0.0), 0 <= tok < 2^32."""
    i = (s.float() + 0.0).view(torch.int32).to(torch.int64)
    i = torch.where(i < 0, i ^ 0x7FFFFFFF, i)          # order-preserving float -> int
    return i * 4294967296 + (M32 - tok.to(torch.int64))


# --------------------------------------------------------------------------- merge
def grouped_merge(state_score, state_pay, g, score, pay, K, pay_fill=None):
    """In-place exact merge of candidates into per-group top-K (higher score is better).

    state_score [G, K] sorted descending; state_pay list of [G, K]; g [n] group ids;
    score [n]; pay list of [n]. Only touched groups are read and written.
    """
    if g.numel() == 0:
        return 0
    dev = g.device
    ug, inv, cnt = torch.unique(g, return_inverse=True, return_counts=True)
    o1 = torch.argsort(score, descending=True, stable=True)
    o2 = torch.argsort(inv[o1], stable=True)
    order = o1[o2]
    inv_s = inv[order]
    start = torch.cumsum(cnt, 0) - cnt
    rank = torch.arange(order.numel(), device=dev) - start[inv_s]
    keep = rank < K  # at most K new entries per group can survive
    nu = ug.numel()
    fill = NEG_INF if score.is_floating_point() else I64_MIN
    cs = torch.full((nu, K), fill, dtype=score.dtype, device=dev)
    cs[inv_s[keep], rank[keep]] = score[order][keep]
    allv = torch.cat((state_score[ug], cs), 1)
    newv, idx = torch.topk(allv, K, dim=1)
    state_score[ug] = newv
    for j, (sp, p) in enumerate(zip(state_pay, pay)):
        pf = pay_fill[j] if pay_fill is not None else (-1 if not p.is_floating_point() else float("nan"))
        cp = torch.full((nu, K), pf, dtype=p.dtype, device=dev)
        cp[inv_s[keep], rank[keep]] = p[order][keep]
        sp[ug] = torch.gather(torch.cat((sp[ug], cp), 1), 1, idx)
    return nu


def bin_of(s: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """s [...], E [..., B] (broadcastable) -> bin index in {-1, 0..B-1}."""
    return (s.unsqueeze(-1) >= E).sum(-1) - 1


# --------------------------------------------------------------------------- config
@dataclass
class AtlasConfig:
    K: int = 64                 # top-K per (layer, pole, neuron)
    Kr: int = 16                # reservoir size per (layer, pole, neuron, bin)
    rho_res: float = 0.01       # rows with u < rho_res*2^32 are fully examined for the reservoir
    rho_sub: float = 0.05       # hashed subsample for moments / enrichment / Gram (>= rho_res)
    enrich_levels: tuple = (2, 3, 4)  # bin indices used as cumulative thresholds s >= E_b
    gram: bool = True
    cap_frac: float = 0.005     # candidate buffer per layer = cap_frac * T * D (overflow -> dense redo, exact)
    cap_frac_early: float = 0.0 # if > 0: buffer for layers < early_layers (layer 0 overflows on single extreme proteins)
    early_layers: int = 2
    backend: str = "torch"      # "torch" | "triton"
    stats_impl: str = "v3"      # subsample statistics: "v3" Triton kernels (CUDA; CPU falls back to v1) | "v2" | "v1" (reference)
    merge_impl: str = "stacked" # reservoir bulk-bin candidates: "stacked" across layers | "loop" (per layer and pole)
    seed_c: float = 4.0         # reservoir seed: expected keys under the seed over the stream = seed_c * Kr
    seed_z: float = 2.0         # ...computed from a z-sigma lower bound on the pilot bin mass
    seed: int = 0
    reservoir: bool = True      # False: top-K only (phase-1 comparison)
    subsample: bool = True      # False: no moments / enrichment / Gram / bin masses
    naive: bool = False         # True: baseline = full per-batch topk merge on every batch (no filter)


@dataclass
class BatchCtx:
    """Per-batch host-known row information (built before the forward)."""
    T: int                       # rows of the flattened [B*L, D] hook output (incl. padding)
    valid_rows: torch.Tensor     # [n_v] int64 positions into the T rows (device)
    tok: torch.Tensor            # [n_v] int64 packed token ids (device)
    key: torch.Tensor            # [n_v] int64 hash keys (device)
    rowmap: torch.Tensor         # [T] int64: valid index of each row, -1 if invalid (device)
    sub_idx: torch.Tensor        # [n_s] indices into valid set with key < rho_sub (device)
    res_in_sub: torch.Tensor     # [n_r] indices into sub_idx with key < rho_res (device)
    ann_sub: torch.Tensor | None = None  # [n_s, n_a] float multi-hot (device) or None
    ann_sub_i8: torch.Tensor | None = None  # same, int8, padded to multiples of 8 for torch._int_mm
    n_v: int = 0
    n_s: int = 0
    n_r: int = 0
    sub_rows: torch.Tensor | None = None  # [n_s] rows of H holding the subsample (device)

    def to(self, device, non_blocking: bool = False) -> "BatchCtx":
        """Copy of this context with every tensor on `device` (sizes are host ints already)."""
        mv = lambda t: None if t is None else t.to(device, non_blocking=non_blocking)  # noqa: E731
        return BatchCtx(T=self.T, valid_rows=mv(self.valid_rows), tok=mv(self.tok), key=mv(self.key), rowmap=mv(self.rowmap),
                        sub_idx=mv(self.sub_idx), res_in_sub=mv(self.res_in_sub), ann_sub=mv(self.ann_sub),
                        ann_sub_i8=mv(self.ann_sub_i8), n_v=self.n_v, n_s=self.n_s, n_r=self.n_r, sub_rows=mv(self.sub_rows))


class Atlas:
    def __init__(self, n_layers: int, D: int, edges: torch.Tensor, cfg: AtlasConfig,
                 n_annot: int = 0, device="cpu"):
        L, B = n_layers, edges.shape[-1]
        assert edges.shape == (L, 2, D, B), edges.shape
        self.L, self.D, self.B, self.cfg = L, D, B, cfg
        self.dev = torch.device(device)
        dev = self.dev
        self.E = edges.to(dev, torch.float32).contiguous()
        self.U_res = int(cfg.rho_res * 2**32)
        self.U_sub = int(cfg.rho_sub * 2**32)
        assert self.U_sub >= self.U_res
        K, Kr = cfg.K, cfg.Kr
        G = L * 2 * D
        self.top_s = torch.full((G, K), NEG_INF, device=dev)
        self.top_t = torch.full((G, K), -1, dtype=torch.long, device=dev)
        self.top_k = torch.full((G, K), I64_MIN, dtype=torch.long, device=dev)  # top_key(s, tok), descending
        self.res_s = torch.full((G * B, Kr), I64_MIN, dtype=torch.long, device=dev)  # = -key
        self.res_t = torch.full((G * B, Kr), -1, dtype=torch.long, device=dev)
        self.res_v = torch.full((G * B, Kr), float("nan"), device=dev)
        # hashed-subsample statistics
        self.n_seen = 0
        self.n_sub = 0
        self.sub_sum = torch.zeros(L, D, dtype=torch.float64, device=dev)
        self.sub_sq = torch.zeros(L, D, dtype=torch.float64, device=dev)
        self.sub_bin = torch.zeros(L, 2, D, B, dtype=torch.int64, device=dev)  # bin masses (counts)
        self.n_a = n_annot
        nl = len(cfg.enrich_levels)
        if n_annot:
            self.enr = torch.zeros(L, 2, nl, D, n_annot, dtype=torch.int64, device=dev)
            self.ann_tot = torch.zeros(n_annot, dtype=torch.int64, device=dev)
        self.gram = torch.zeros(L, D, D, dtype=torch.float32, device=dev) if cfg.gram else None
        self.gram64 = torch.zeros(L, D, D, dtype=torch.float64) if cfg.gram else None  # host flush
        self.gram_rows = 0  # subsample rows in the Gram (not checkpointed: written once at the end)
        # seeds (exact-preserving thresholds; see seed()). Unseeded = no-op values.
        self.top_seed = torch.full((G,), NEG_INF, device=dev)              # admit s > top_seed
        self.res_seed = torch.full((G * B,), I64_MIN, dtype=torch.long, device=dev)  # admit -key > res_seed
        self.enabled = True  # False: hooks are no-ops (forward-only timing)
        self.stats = {"batches": 0, "dense_layer_batches": 0, "redo_layers": 0,
                      "compacted": 0, "elements": 0, "top_cand": 0, "res_cand": 0}
        self.ctx: BatchCtx | None = None
        self._pending: dict[int, tuple] = {}
        self._dense_layers: set[int] = set(range(L))  # first batch is dense (warm-up)
        self._active: set[int] | None = None
        self._kernel = None
        if cfg.backend == "triton":
            from .filter_triton import filter_candidates  # noqa: F401
            self._kernel = filter_candidates

    # ------------------------------------------------------------------ batch setup
    def make_ctx(self, T: int, valid_rows: torch.Tensor, tok: torch.Tensor,
                 ann_ids: torch.Tensor | None = None, device=None) -> BatchCtx:
        """Host tensors in; everything row-level is decided here (no device sync later).
        ann_ids: [n_v, M_a] int64 annotation ids per valid token (-1 padded) or None."""
        assert tok.numel() == 0 or (int(tok.min()) >= 0 and int(tok.max()) < 2**32), "token ids must be in [0, 2^32)"
        key = token_keys(tok, self.cfg.seed)
        sub = torch.nonzero(key < self.U_sub).squeeze(1)
        res_in_sub = torch.nonzero(key[sub] < self.U_res).squeeze(1)
        rowmap = torch.full((T,), -1, dtype=torch.long)
        rowmap[valid_rows] = torch.arange(valid_rows.numel())
        ann_sub = None
        if self.n_a and ann_ids is not None:
            a = ann_ids[sub]
            ann_sub = torch.zeros(sub.numel(), self.n_a)
            r, c = torch.nonzero(a >= 0, as_tuple=True)
            ann_sub[r, a[r, c]] = 1.0
        d = self.dev if device is None else torch.device(device)  # "cpu": host-only (e.g. a prefetch thread), then ctx.to(dev)
        ann_i8 = None
        if ann_sub is not None:
            r8, c8 = -(-ann_sub.shape[0] // 8) * 8, -(-ann_sub.shape[1] // 8) * 8
            ann_i8 = torch.zeros(r8, c8, dtype=torch.int8)
            ann_i8[: ann_sub.shape[0], : ann_sub.shape[1]] = ann_sub.to(torch.int8)
            ann_i8 = ann_i8.to(d)
        return BatchCtx(T=T, valid_rows=valid_rows.to(d), tok=tok.to(d), key=key.to(d),
                        rowmap=rowmap.to(d), sub_idx=sub.to(d), res_in_sub=res_in_sub.to(d),
                        ann_sub=None if ann_sub is None else ann_sub.to(d), ann_sub_i8=ann_i8,
                        n_v=valid_rows.numel(), n_s=sub.numel(), n_r=res_in_sub.numel(),
                        sub_rows=valid_rows[sub].to(d))

    def begin_batch(self, ctx: BatchCtx, redo_layers: list[int] | None = None):
        """redo_layers=None: a fresh batch over all layers. Otherwise a re-run of the same
        batch where only the listed (overflowed) layers are processed, densely."""
        self.ctx = ctx
        self._pending = {}
        if redo_layers is None:
            self._active = set(range(self.L))
            self._sub_done = set()
            self.stats["elements"] += 2 * ctx.n_v * self.D * self.L
        else:
            self._active = set(redo_layers)
        # thresholds for the compaction pass, from state at batch start (device, no sync)
        L, D, B, Kr = self.L, self.D, self.B, self.cfg.Kr
        # s >= tau_K (ties at tau can still enter with a smaller id)  <=>  s > prev(tau_K); and s > seed
        tau = self.top_s[:, -1]
        tau_top = torch.maximum(torch.nextafter(tau, torch.full_like(tau, NEG_INF)), self.top_seed).view(L, 2, D)
        kth = torch.maximum(self.res_s[:, Kr - 1], self.res_seed).view(L, 2, D, B)
        underfilled = kth < -self.U_res  # key threshold > U_res (or reservoir not full)
        eps = torch.where(underfilled, self.E, torch.full_like(self.E, float("inf"))).amin(-1)
        eps = torch.nextafter(eps, torch.full_like(eps, NEG_INF))  # s >= E  <=>  s > prev(E)
        self.thr = torch.minimum(tau_top, eps) if self.cfg.reservoir else tau_top.clone()  # [L, 2, D]

    # ------------------------------------------------------------------ hook body
    @torch.no_grad()
    def observe(self, l: int, H: torch.Tensor):
        """H: [T, D] hook output for layer l (any float dtype, rows include padding)."""
        if not self.enabled or l not in self._active:
            return
        ctx = self.ctx
        if l in self._dense_layers:
            with _rf("atlas.dense"):
                self._dense(l, H)
            return
        # 1) one compare-and-compact pass over H
        with _rf("atlas.compact"):
            if self._kernel is not None:
                vals, rows, cols, cnt = self._kernel(H, ctx.rowmap, self.thr[l, 0], self.thr[l, 1],
                                                     self._cap(ctx, l))
            else:  # plain nonzero: one sync, but faster than nonzero_static on CUDA (phase 1)
                valid = ctx.rowmap >= 0
                mask = ((H > self.thr[l, 0]) | (H < -self.thr[l, 1])) & valid[:, None]
                idx = torch.nonzero(mask)
                cnt = idx.shape[0]  # host int
                rows, cols = idx[:, 0], idx[:, 1]
                vals = H[rows, cols].float()
                rows = ctx.rowmap[rows]
        # 2) hashed subsample rows (host-known sizes)
        R = None
        if self.cfg.subsample or self.cfg.reservoir:
            with _rf("atlas.subsample"):
                Hn = H.index_select(0, ctx.sub_rows)  # [n_s, D] native dtype
                if self.cfg.subsample and l not in self._sub_done:
                    self._subsample_stats(l, Hn)
                R = Hn.index_select(0, ctx.res_in_sub).float() if self.cfg.reservoir else None
        self._pending[l] = (vals, rows, cols, cnt, R)

    def _cap(self, ctx, l=None):
        f = self.cfg.cap_frac
        if self.cfg.cap_frac_early > 0 and l is not None and l < self.cfg.early_layers:
            f = self.cfg.cap_frac_early
        return max(4096, int(f * ctx.n_v * self.D))

    def _subsample_stats(self, l, Hn):
        """Hn [n_s, D]: the subsample rows in the hook's dtype."""
        impl = self.cfg.stats_impl
        if impl == "v3" and self.dev.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
            impl = "v1"  # Triton kernels need CUDA (or the interpreter, for tests); v1 computes the same quantities
        if impl == "v3":
            return self._subsample_stats_v3(l, Hn)
        if impl == "v2":
            return self._subsample_stats_v2(l, Hn.float(), Hn)
        return self._subsample_stats_v1(l, Hn.float())

    def _subsample_stats_v1(self, l, Hs):
        self._sub_done.add(l)
        if Hs.shape[0] == 0:
            return
        # fp32 within the batch (n_s ~ 1.6k rows), fp64 only for the [D] running sums
        with _rf("atlas.sub.moments"):
            self.sub_sum[l] += Hs.sum(0).double()
            self.sub_sq[l] += (Hs * Hs).sum(0).double()
        levels = list(self.cfg.enrich_levels)
        with _rf("atlas.sub.bins_enrich"):
            for p, sgn in ((0, 1.0), (1, -1.0)):
                s = Hs * sgn
                ge = s.unsqueeze(-1) >= self.E[l, p]           # [n_s, D, B] cumulative thresholds
                cnt_ge = ge.sum(0)                              # [D, B] int64, no atomics
                nxt = torch.cat((cnt_ge[:, 1:], torch.zeros_like(cnt_ge[:, :1])), 1)
                self.sub_bin[l, p] += cnt_ge - nxt              # exact bin counts
                if self.n_a and self.ctx.ann_sub is not None:
                    m = ge[:, :, levels].permute(0, 2, 1).reshape(Hs.shape[0], -1).float()  # [n_s, nl*D]
                    with _tf32():  # 0/1 inputs are exact in TF32; per-batch counts < 2^24
                        c = m.T @ self.ctx.ann_sub              # [nl*D, n_a]
                    self.enr[l, p] += c.view(len(levels), self.D, -1).round().to(torch.int64)
        if self.gram is not None:
            with _rf("atlas.sub.gram"):
                with _tf32():
                    self.gram[l] += Hs.T @ Hs
                if l == 0:
                    self.gram_rows += Hs.shape[0]

    def _subsample_stats_v2(self, l, Hs, Hs_native=None):
        """Same quantities as v1, without materialising [n_s, D, B] booleans or permuted copies:
        one transposed copy, bin index by searchsorted (negative pole against flipped -E, no
        negated copy of H), counts by integer scatter_add, enrichment by an exact integer GEMM
        on masks read off the bin index, Gram on native-dtype inputs with fp32 output."""
        self._sub_done.add(l)
        n = Hs.shape[0]
        if n == 0:
            return
        D, B = self.D, self.B
        with _rf("atlas.sub.moments"):
            self.sub_sum[l] += Hs.sum(0).double()
            self.sub_sq[l] += (Hs * Hs).sum(0).double()
        with _rf("atlas.sub.bins"):
            HT = Hs.T.contiguous()                                           # [D, n]
            b0 = torch.searchsorted(self.E[l, 0], HT, right=True) - 1        # #{E0_b <= h} - 1
            F = torch.flip(-self.E[l, 1], dims=(-1,)).contiguous()          # ascending
            b1 = B - torch.searchsorted(F, HT, right=False) - 1              # #{-E1_b >= h} - 1
            base = (torch.arange(D, device=self.dev) * (B + 1))[:, None]
            acc = torch.zeros(2, D * (B + 1), dtype=torch.int32, device=self.dev)
            ones = torch.ones(D * n, dtype=torch.int32, device=self.dev)
            for p, bb in ((0, b0), (1, b1)):
                acc[p].scatter_add_(0, (base + bb + 1).reshape(-1), ones)    # slot 0 = "not in pole"
            self.sub_bin[l] += acc.view(2, D, B + 1)[:, :, 1:].to(torch.int64)
        if self.n_a and self.ctx.ann_sub is not None:
            with _rf("atlas.sub.enrich"):
                levels = list(self.cfg.enrich_levels)
                for p, bb in ((0, b0), (1, b1)):
                    m = torch.cat([(bb >= lev) for lev in levels], 0)        # [nl*D, n] bool, contiguous
                    c = _count_mm(m, self.ctx.ann_sub_i8, self.ctx.ann_sub)  # [nl*D, n_a] exact int
                    self.enr[l, p] += c[:, : self.n_a].view(len(levels), D, self.n_a).to(torch.int64)
        if self.gram is not None:
            with _rf("atlas.sub.gram"):
                X = Hs_native if Hs_native is not None else Hs
                self.gram[l] += _mm_f32out(X.T, X)
                if l == 0:
                    self.gram_rows += n

    def _subsample_stats_v3(self, l, Hn):
        """Two Triton kernels (atlas/stats_triton.py): one pass over the subsample rows for bin
        counts, moments and int8 enrichment masks, then one exact int8 GEMM for the contingency
        counts; Gram on the upper block triangle, accumulated in place."""
        from .stats_triton import gram_accumulate, sub_stats
        self._sub_done.add(l)
        n = Hn.shape[0]
        if n == 0:
            return
        D = self.D
        ann = self.n_a and self.ctx.ann_sub is not None
        levels = list(self.cfg.enrich_levels) if ann else []
        with _rf("atlas.sub.kernel"):
            s, sq, cnt, M = sub_stats(Hn, self.E[l, 0], self.E[l, 1], levels)
            self.sub_sum[l] += s.double()
            self.sub_sq[l] += sq.double()
            self.sub_bin[l] += cnt.to(torch.int64)
        if ann:
            with _rf("atlas.sub.enrich"):
                NL = len(levels)
                c = _count_mm_i8(M.reshape(2 * NL * D, M.shape[-1]), self.ctx.ann_sub_i8, self.ctx.ann_sub)
                self.enr[l] += c[:, : self.n_a].reshape(2, NL, D, self.n_a).to(torch.int64)
        if self.gram is not None:
            with _rf("atlas.sub.gram"):
                gram_accumulate(Hn, self.gram[l])
                if l == 0:
                    self.gram_rows += n

    # ------------------------------------------------------------------ dense path
    def _dense(self, l, H):
        """Warm-up / overflow path: exact, sync-heavy, no thresholds used."""
        ctx = self.ctx
        Hv = H.index_select(0, ctx.valid_rows)  # native dtype (bf16 topk is exact)
        if self.cfg.subsample and l not in self._sub_done:  # once per layer per batch
            self._subsample_stats(l, Hv.index_select(0, ctx.sub_idx))
        D, B, K, Kr = self.D, self.B, self.cfg.K, self.cfg.Kr
        cols = torch.arange(D, device=self.dev)
        for p, sgn in ((0, 1.0), (1, -1.0)):
            S = Hv if p == 0 else -Hv
            k = min(K, S.shape[0])
            if k > 0:
                vk = torch.topk(S, k, dim=0).values[k - 1]               # K-th value per column (S's dtype: exact)
                gD = (l * 2 + p) * D + cols
                r, c = torch.nonzero(S >= vk[None, :], as_tuple=True)   # includes every row tied at the K-th value
                v = S[r, c].float()
                ok = v > self.top_seed[gD[c]]
                r, c, v = r[ok], c[ok], v[ok]
                t = ctx.tok[r]
                self.stats["top_cand"] += int(v.numel())
                grouped_merge(self.top_k, [self.top_t, self.top_s], gD[c], top_key(v, t), [t, v], K,
                              pay_fill=[-1, NEG_INF])
            if not self.cfg.reservoir:
                continue
            S = S.float()
            b = bin_of(S, self.E[l, p]).to(torch.int8)  # [n_v, D]
            gb = (((l * 2 + p) * D + cols) * B)  # [D] base group ids
            for bb in range(B):
                seed_key = -self.res_seed[gb + bb]  # admit key < seed_key (I64_MIN -> no seed)
                inb = (b == bb) & ((self.res_seed[gb + bb] == I64_MIN)[None, :] | (ctx.key[:, None] < seed_key[None, :]))
                km = torch.where(inb, ctx.key[:, None], torch.full(b.shape, torch.iinfo(torch.int64).max, dtype=torch.long, device=b.device))
                kr = min(Kr, km.shape[0])
                kv, ki = torch.topk(km, kr, dim=0, largest=False)
                ok = kv < torch.iinfo(torch.int64).max
                gg = (((l * 2 + p) * D + cols) * B + bb).expand(kr, D)[ok]
                self.stats["res_cand"] += gg.numel()
                grouped_merge(self.res_s, [self.res_t, self.res_v], gg, -kv[ok],
                              [ctx.tok[ki[ok]], Hv[ki[ok], cols.expand(kr, D)[ok]].float()], Kr)
        self.stats["dense_layer_batches"] += 1
        self._pending[l] = None

    # ------------------------------------------------------------------ after forward
    @torch.no_grad()
    def end_batch(self) -> list[int]:
        """Merge pending candidates. Returns layers that overflowed (caller re-runs the
        forward with those layers in dense mode; nothing was merged for them)."""
        ctx = self.ctx
        filt = [l for l, v in self._pending.items() if v is not None]
        redo = []
        if filt:
            dev_cnt = [l for l in filt if torch.is_tensor(self._pending[l][3])]
            got = dict(zip(dev_cnt, torch.stack([self._pending[l][3].reshape(()) for l in dev_cnt]).tolist())) if dev_cnt else {}
            cnts = [got.get(l, self._pending[l][3]) for l in filt]  # sync #1 (Triton counts)
            redo = [l for l, c in zip(filt, cnts) if c > self._cap(ctx, l)]
            ok = [(l, c) for l, c in zip(filt, cnts) if c <= self._cap(ctx, l)]
            if ok:
                self._merge_filtered(ok)
        dense_done = [l for l, v in self._pending.items() if v is None]
        if not self.cfg.naive:
            self._dense_layers -= set(dense_done)
        if redo:
            self.stats["redo_layers"] += len(redo)
            self._dense_layers |= set(redo)
        else:
            self.n_seen += ctx.n_v
            self.n_sub += ctx.n_s
            if self.n_a and ctx.ann_sub is not None:
                self.ann_tot += ctx.ann_sub.sum(0).round().to(torch.int64)
            self.stats["batches"] += 1
        return redo

    def _merge_filtered(self, ok):
        with _rf("atlas.merge"):
            self._merge_filtered_impl(ok)

    def _merge_filtered_impl(self, ok):
        ctx, D, B, K, Kr = self.ctx, self.D, self.B, self.cfg.K, self.cfg.Kr
        dev = self.dev
        parts = [(torch.full((c,), l, device=dev, dtype=torch.long),) + tuple(t[:c] for t in self._pending[l][:3])
                 for l, c in ok]
        lay = torch.cat([p[0] for p in parts])
        vals = torch.cat([p[1] for p in parts])
        rows = torch.cat([p[2] for p in parts])
        cols = torch.cat([p[3] for p in parts])
        self.stats["compacted"] += lay.numel()
        tok, key = ctx.tok[rows], ctx.key[rows]
        kth_key = self.top_k[:, K - 1]
        res_thr = torch.maximum(self.res_s[:, Kr - 1], self.res_seed)
        top_g, top_v, top_t = [], [], []
        res_g, res_v, res_k, res_t = [], [], [], []
        with _rf("atlas.merge.cand"):
            for p, sgn in ((0, 1.0), (1, -1.0)):
                s = vals * sgn
                g = (lay * 2 + p) * D + cols
                m = (s > self.top_seed[g]) & (top_key(s, tok) > kth_key[g])
                top_g.append(g[m]); top_v.append(s[m]); top_t.append(tok[m])
                if not self.cfg.reservoir:
                    continue
                b = bin_of(s, self.E[lay, p, cols])
                gr = g * B + b.clamp(min=0)
                m = (b >= 0) & (key >= self.U_res) & (-key > res_thr[gr])
                res_g.append(gr[m]); res_k.append(key[m]); res_t.append(tok[m]); res_v.append(vals[m])
        # bulk bins: every element of the rows with key < U_res
        if self.cfg.reservoir and self.cfg.merge_impl == "stacked":
            with _rf("atlas.merge.bulk"):
                self._bulk_stacked(ok, res_thr, res_g, res_k, res_t, res_v)
        if self.cfg.reservoir and self.cfg.merge_impl == "loop":
            with _rf("atlas.merge.bulk"):
                self._bulk_loop(ok, res_thr, res_g, res_k, res_t, res_v)
        with _rf("atlas.merge.top"):
            tg, tv, tt = torch.cat(top_g), torch.cat(top_v), torch.cat(top_t)
            self.stats["top_cand"] += tg.numel()
            grouped_merge(self.top_k, [self.top_t, self.top_s], tg, top_key(tv, tt), [tt, tv], K,
                          pay_fill=[-1, NEG_INF])
        if not self.cfg.reservoir:
            return
        with _rf("atlas.merge.res"):
            rg = torch.cat(res_g)
            self.stats["res_cand"] += rg.numel()
            grouped_merge(self.res_s, [self.res_t, self.res_v], rg, -torch.cat(res_k),
                          [torch.cat(res_t), torch.cat(res_v)], Kr)

    def _bulk_loop(self, ok, res_thr, res_g, res_k, res_t, res_v):
        """Reference: reservoir candidates from the rows with key < U_res, per layer and pole."""
        ctx, D, B, dev = self.ctx, self.D, self.B, self.dev
        for l, _ in ok:
            R = self._pending[l][4]  # [n_r, D] float
            if R is None or R.shape[0] == 0:
                continue
            ridx = ctx.sub_idx.index_select(0, ctx.res_in_sub)  # valid-set indices
            rk, rt = ctx.key[ridx], ctx.tok[ridx]
            colsD = torch.arange(D, device=dev)
            for p, sgn in ((0, 1.0), (1, -1.0)):
                s = R * sgn
                b = bin_of(s, self.E[l, p])  # [n_r, D]
                gr = (((l * 2 + p) * D + colsD) * B)[None, :] + b.clamp(min=0)
                m = (b >= 0) & (-rk[:, None] > res_thr[gr])
                r_i, c_i = torch.nonzero(m, as_tuple=True)
                res_g.append(gr[r_i, c_i]); res_k.append(rk[r_i]); res_t.append(rt[r_i]); res_v.append(R[r_i, c_i])

    def _bulk_stacked(self, ok, res_thr, res_g, res_k, res_t, res_v, max_elems: int = 16_000_000):
        """Reservoir candidates from the rows with key < U_res, stacked across layers: one
        searchsorted and one nonzero per layer group instead of one per layer and pole. Exact
        prefilter: a row can only enter a bin whose key ceiling exceeds its key, so rows with key
        >= the loosest live ceiling (over the layers in this batch) are dropped first."""
        ctx, D, B, dev = self.ctx, self.D, self.B, self.dev
        Ls = [l for l, _ in ok if self._pending[l][4] is not None and self._pending[l][4].shape[0] > 0]
        if not Ls:
            return
        ridx = ctx.sub_idx.index_select(0, ctx.res_in_sub)
        rk, rt = ctx.key[ridx], ctx.tok[ridx]
        Lt = torch.tensor(Ls, device=dev)
        loosest = res_thr.view(self.L, 2 * D * B).index_select(0, Lt).min()
        if int(loosest) != I64_MIN:  # every bin has a ceiling: keep rows under the loosest one
            keep = torch.nonzero(-rk > loosest).squeeze(1)
        else:
            keep = None
        if keep is not None and keep.numel() == 0:
            self.stats["bulk_rows_in"] = self.stats.get("bulk_rows_in", 0) + rk.numel()
            return
        rk_k = rk if keep is None else rk[keep]
        rt_k = rt if keep is None else rt[keep]
        nk = rk_k.numel()
        self.stats["bulk_rows_in"] = self.stats.get("bulk_rows_in", 0) + rk.numel()  # prefilter effect
        self.stats["bulk_rows_kept"] = self.stats.get("bulk_rows_kept", 0) + nk
        colsD = torch.arange(D, device=dev)
        per = max(1, max_elems // max(1, D * nk))
        for g0 in range(0, len(Ls), per):
            Lg = Ls[g0:g0 + per]
            Lgt = torch.tensor(Lg, device=dev)
            R = torch.stack([self._pending[l][4] for l in Lg])            # [nL, n_r, D]
            if keep is not None:
                R = R.index_select(1, keep)
            HT = R.transpose(1, 2).contiguous()                             # [nL, D, nk]
            E = self.E.index_select(0, Lgt)                                 # [nL, 2, D, B]
            b0 = torch.searchsorted(E[:, 0].contiguous(), HT, right=True) - 1
            F = torch.flip(-E[:, 1], dims=(-1,)).contiguous()
            b1 = B - torch.searchsorted(F, HT, right=False) - 1
            for p, bb in ((0, b0), (1, b1)):
                gbase = ((Lgt[:, None] * 2 + p) * D + colsD[None, :]) * B   # [nL, D]
                gr = gbase[:, :, None] + bb.clamp(min=0)                    # [nL, D, nk]
                m = (bb >= 0) & (-rk_k[None, None, :] > res_thr[gr])
                li, di, ri = torch.nonzero(m, as_tuple=True)
                res_g.append(gr[li, di, ri]); res_k.append(rk_k[ri]); res_t.append(rt_k[ri]); res_v.append(HT[li, di, ri])

    # ------------------------------------------------------------------ outputs
    def flush_gram(self):
        """Move the fp32 device Gram into the fp64 host accumulator. Copies fp32 (half the bytes
        of widening on the device first) and widens on the host; a full flush moves L*D^2*4 bytes
        (3.5 GB for 650M), so callers flush rarely (run_atlas: --gram-flush-every)."""
        if self.gram is not None:
            self.gram64 += self.gram.cpu().double()
            self.gram.zero_()

    # ------------------------------------------------------------------ checkpoint / merge
    _STATE = ["top_s", "top_t", "top_k", "res_s", "res_t", "res_v", "sub_sum", "sub_sq", "sub_bin", "E",
              "top_seed", "res_seed"]

    def state_dict(self) -> dict:
        """Everything needed to continue the stream exactly (CPU tensors). The Gram is NOT
        included (subsample structure, written once at the end); a resumed run's Gram covers
        the batches after the resume and says so via gram_rows."""
        st = {k: getattr(self, k).cpu() for k in self._STATE}
        if self.n_a:
            st["enr"], st["ann_tot"] = self.enr.cpu(), self.ann_tot.cpu()
        st.update(n_seen=self.n_seen, n_sub=self.n_sub, stats=dict(self.stats),
                  dense_layers=sorted(self._dense_layers), cfg=dict(self.cfg.__dict__), n_a=self.n_a)
        return st

    def load_state_dict(self, st: dict):
        assert st["cfg"] == dict(self.cfg.__dict__), "config mismatch on resume"
        assert torch.equal(st["E"].to(self.E.device), self.E), "bin edges differ: resume must reuse the checkpointed pilot"
        for k in self._STATE:
            getattr(self, k).copy_(st[k].to(self.dev))
        if self.n_a:
            self.enr.copy_(st["enr"].to(self.dev)); self.ann_tot.copy_(st["ann_tot"].to(self.dev))
        if self.gram is not None:
            self.gram64.zero_(); self.gram.zero_(); self.gram_rows = 0
        self.n_seen, self.n_sub, self.stats = st["n_seen"], st["n_sub"], dict(st["stats"])
        self._dense_layers = set(st["dense_layers"])

    @torch.no_grad()
    def merge_from(self, other: "Atlas"):
        """Fold in another Atlas run over a DISJOINT set of tokens (same edges, cfg, seed).
        Top-K and reservoirs: concat + select per group (exact); subsample stats: sums."""
        assert torch.equal(other.E.to(self.dev), self.E) and other.cfg.__dict__ == self.cfg.__dict__
        assert torch.equal(other.top_seed.to(self.dev), self.top_seed) and torch.equal(other.res_seed.to(self.dev), self.res_seed), \
            "shards must share seeds (derive them from the GLOBAL stream size)"
        K, Kr = self.cfg.K, self.cfg.Kr
        v, i = torch.topk(torch.cat((self.top_k, other.top_k.to(self.dev)), 1), K, dim=1)  # keys are unique
        self.top_t = torch.gather(torch.cat((self.top_t, other.top_t.to(self.dev)), 1), 1, i)
        self.top_s = torch.gather(torch.cat((self.top_s, other.top_s.to(self.dev)), 1), 1, i)
        self.top_k = v
        v, i = torch.topk(torch.cat((self.res_s, other.res_s.to(self.dev)), 1), Kr, dim=1)
        self.res_t = torch.gather(torch.cat((self.res_t, other.res_t.to(self.dev)), 1), 1, i)
        self.res_v = torch.gather(torch.cat((self.res_v, other.res_v.to(self.dev)), 1), 1, i)
        self.res_s = v
        for k in ["sub_sum", "sub_sq", "sub_bin"]:
            getattr(self, k).add_(getattr(other, k).to(self.dev))
        if self.n_a:
            self.enr += other.enr.to(self.dev); self.ann_tot += other.ann_tot.to(self.dev)
        if self.gram is not None:
            self.flush_gram(); other.flush_gram(); self.gram64 += other.gram64
            self.gram_rows += other.gram_rows
        self.n_seen += other.n_seen
        self.n_sub += other.n_sub
        for k, x in other.stats.items():
            self.stats[k] = self.stats.get(k, 0) + x

    def result(self) -> dict:
        L, D, B, K, Kr = self.L, self.D, self.B, self.cfg.K, self.cfg.Kr
        self.flush_gram()
        top_s = self.top_s.view(L, 2, D, K).cpu()
        sign = torch.tensor([1.0, -1.0]).view(1, 2, 1, 1)
        out = {
            "top_val": top_s * sign, "top_tok": self.top_t.view(L, 2, D, K).cpu(),
            "res_tok": self.res_t.view(L, 2, D, B, Kr).cpu(), "res_val": self.res_v.view(L, 2, D, B, Kr).cpu(),
            "res_key": -self.res_s.view(L, 2, D, B, Kr).cpu(),
            "edges": self.E.cpu(), "n_seen": self.n_seen, "n_sub": self.n_sub,
            "sub_sum": self.sub_sum.cpu(), "sub_sq": self.sub_sq.cpu(), "sub_bin": self.sub_bin.cpu(),
            "stats": dict(self.stats), "cfg": self.cfg.__dict__,
        }
        if self.n_a:
            out["enr"] = self.enr.cpu()
            out["ann_tot"] = self.ann_tot.cpu()
        if self.gram is not None:
            out["gram"] = symmetrize_upper(self.gram64)  # v3 accumulates the upper block triangle only
            out["gram_rows"] = self.gram_rows
        out.update(self.verify_seeds())
        return out

    # ------------------------------------------------------------------ seeds
    @torch.no_grad()
    def seed(self, top_seed: torch.Tensor | None = None, res_seed_key: torch.Tensor | None = None):
        """Exact-preserving start thresholds.
        top_seed [L,2,D]: admit s > top_seed. Exact when the tokens it was computed from are
          part of the stream (the final K-th value can only be >= it); use prev(pilot K-th).
        res_seed_key [L,2,D,B] int64: admit key < seed (>= 2^32 means unseeded). Bottom-k is
          order-independent, so the result equals the unseeded one for every bin that ends
          with >= Kr keys under its seed; verify_seeds() reports the bins that do not."""
        if top_seed is not None:
            self.top_seed.copy_(top_seed.reshape(-1).to(self.dev, torch.float32))
            self._dense_layers = set()  # no dense warm-up needed for top-K
        if res_seed_key is not None:
            k = res_seed_key.reshape(-1).to(self.dev, torch.long)
            self.res_seed.copy_(torch.where(k >= 2**32, torch.full_like(k, I64_MIN), -k))

    def verify_seeds(self) -> dict:
        """Bins whose reservoir holds < Kr entries although a seed was active: there the result is
        'all keys under the seed' (still a uniform sample), not the unseeded bottom-Kr."""
        n_kept = (self.res_t >= 0).sum(1)
        seeded = self.res_seed != I64_MIN
        bad = seeded & (n_kept < self.cfg.Kr)
        L, D, B = self.L, self.D, self.B
        seed_key = torch.where(seeded, -self.res_seed, torch.full_like(self.res_seed, 2**32))
        return {"res_seeded_bins": int(seeded.sum()), "res_seed_limited_bins": int(bad.sum()),
                "res_seed_limited": bad.view(L, 2, D, B).cpu(), "res_seed_key": seed_key.view(L, 2, D, B).cpu(),
                "top_seed_short_groups": int(((self.top_t >= 0).sum(1) < self.cfg.K).logical_and(self.top_seed > NEG_INF).sum())}


NONZERO_STATIC = {"ok": None}


def _nonzero_static(mask, cap):
    """Sync-free nonzero with fixed output size (fill -1). Falls back to nonzero (one sync) if
    this torch build lacks a kernel for the device; the fallback is recorded."""
    if NONZERO_STATIC["ok"] is not False:
        try:
            out = torch.nonzero_static(mask, size=cap, fill_value=-1)
            NONZERO_STATIC["ok"] = True
            return out
        except (RuntimeError, NotImplementedError, AttributeError) as e:
            NONZERO_STATIC["ok"] = False
            NONZERO_STATIC["error"] = str(e)[:200]
    idx = torch.nonzero(mask)[:cap]
    pad = torch.full((cap - idx.shape[0], 2), -1, dtype=idx.dtype, device=idx.device)
    return torch.cat((idx, pad))


def _rf(name):
    return torch.profiler.record_function(name)


def gemm_paths() -> dict:
    """Which exact-GEMM and nonzero paths this process actually took (None = not exercised)."""
    return {"int_mm": dict(_INT_MM), "mm_out_dtype": dict(_MM_OUT_DTYPE), "nonzero_static": dict(NONZERO_STATIC)}


_MM_OUT_DTYPE = {"ok": None}


def _mm_f32out(a, b):
    """a @ b with fp32 output. bf16/fp16 inputs: one GEMM with fp32 output if this torch build
    supports out_dtype (products of bf16 values are exact in fp32), else the TF32 path on fp32
    copies (also exact products for bf16-origin values). fp32 inputs: TF32 GEMM."""
    if a.dtype in (torch.bfloat16, torch.float16) and a.is_cuda and _MM_OUT_DTYPE["ok"] is not False:
        try:
            out = torch.mm(a, b, out_dtype=torch.float32)
            _MM_OUT_DTYPE["ok"] = True
            return out
        except (TypeError, RuntimeError, NotImplementedError):
            _MM_OUT_DTYPE["ok"] = False
    with _tf32():
        return a.float() @ b.float()


_INT_MM = {"ok": None}


def symmetrize_upper(U: torch.Tensor) -> torch.Tensor:
    """Full symmetric matrix from one whose upper triangle (incl. diagonal) is right. Linear and
    idempotent on symmetric inputs, so full (v1/v2) and upper-only (v3) Grams can be mixed."""
    return torch.triu(U) + torch.triu(U, 1).transpose(-1, -2)


def _count_mm_i8(m_i8, ann_i8, ann_f):
    """Exact counts m_i8[M, n8] (0/1 int8, padded columns zero) @ ann[n, A] -> int [M, >= A]."""
    if m_i8.is_cuda and _INT_MM["ok"] is not False:
        try:
            out = torch._int_mm(m_i8, ann_i8)
            _INT_MM["ok"] = True
            return out
        except (RuntimeError, NotImplementedError, AttributeError) as e:
            _INT_MM["ok"] = False
            _INT_MM["error"] = str(e)[:200]
    n = ann_f.shape[0]
    with _tf32():
        return (m_i8[:, :n].float() @ ann_f).round().to(torch.int32)


def _count_mm(mask, ann_i8, ann_f):
    """Exact co-occurrence counts mask[M, n] (bool) @ ann[n, A] (0/1) -> int [M, A_pad].
    CUDA: torch._int_mm on int8 (int32 accumulate, exact); else fp32 GEMM (exact below 2^24)."""
    M, n = mask.shape
    if mask.is_cuda and _INT_MM["ok"] is not False:
        try:
            n8 = ann_i8.shape[0]
            m8 = -(-M // 8) * 8
            a = torch.zeros(m8, n8, dtype=torch.int8, device=mask.device)
            a[:M, :n] = mask.to(torch.int8)
            out = torch._int_mm(a, ann_i8)[:M]
            _INT_MM["ok"] = True
            return out
        except (RuntimeError, NotImplementedError, AttributeError) as e:
            _INT_MM["ok"] = False
            _INT_MM["error"] = str(e)[:200]
    with _tf32():
        return (mask.float() @ ann_f).round().to(torch.int32)


class _tf32:
    """bf16-origin values are exact in TF32, so TF32 GEMMs give exact products with fp32 accumulation."""

    def __enter__(self):
        self.prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True

    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32 = self.prev
