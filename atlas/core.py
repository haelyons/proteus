"""Streaming per-neuron atlas: exact top-K, exact quantile-stratified bottom-k
reservoirs, and hashed-subsample statistics (moments, enrichment, Gram).

Everything is exact w.r.t. a stated definition:

* top-K   : for every (layer, pole, neuron) the K largest pole-oriented values
            s = +h (pole 0) / -h (pole 1) over all valid tokens. Exact because the
            running K-th value tau only rises (monotone-threshold filter).
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


# --------------------------------------------------------------------------- merge
def grouped_merge(state_score, state_pay, g, score, pay, K):
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
    for sp, p in zip(state_pay, pay):
        pf = -1 if not p.is_floating_point() else float("nan")
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
    backend: str = "torch"      # "torch" | "triton"
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
    n_v: int = 0
    n_s: int = 0
    n_r: int = 0


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
                 ann_ids: torch.Tensor | None = None) -> BatchCtx:
        """Host tensors in; everything row-level is decided here (no device sync later).
        ann_ids: [n_v, M_a] int64 annotation ids per valid token (-1 padded) or None."""
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
        d = self.dev
        return BatchCtx(T=T, valid_rows=valid_rows.to(d), tok=tok.to(d), key=key.to(d),
                        rowmap=rowmap.to(d), sub_idx=sub.to(d), res_in_sub=res_in_sub.to(d),
                        ann_sub=None if ann_sub is None else ann_sub.to(d),
                        n_v=valid_rows.numel(), n_s=sub.numel(), n_r=res_in_sub.numel())

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
        tau_top = torch.maximum(self.top_s[:, -1], self.top_seed).view(L, 2, D)
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
                                                     self._cap(ctx))
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
                Hs = H.index_select(0, ctx.valid_rows.index_select(0, ctx.sub_idx)).float()  # [n_s, D]
                if self.cfg.subsample and l not in self._sub_done:
                    self._subsample_stats(l, Hs)
                R = Hs.index_select(0, ctx.res_in_sub) if self.cfg.reservoir else None
        self._pending[l] = (vals, rows, cols, cnt, R)

    def _cap(self, ctx):
        return max(4096, int(self.cfg.cap_frac * ctx.n_v * self.D))

    def _subsample_stats(self, l, Hs):
        self._sub_done.add(l)
        if Hs.shape[0] == 0:
            return
        # fp32 within the batch (n_s ~ 1.6k rows), fp64 only for the [D] running sums
        self.sub_sum[l] += Hs.sum(0).double()
        self.sub_sq[l] += (Hs * Hs).sum(0).double()
        levels = list(self.cfg.enrich_levels)
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
            with _tf32():
                self.gram[l] += Hs.T @ Hs
            if l == 0:
                self.gram_rows += Hs.shape[0]

    # ------------------------------------------------------------------ dense path
    def _dense(self, l, H):
        """Warm-up / overflow path: exact, sync-heavy, no thresholds used."""
        ctx = self.ctx
        Hv = H.index_select(0, ctx.valid_rows)  # native dtype (bf16 topk is exact)
        if self.cfg.subsample and l not in self._sub_done:  # once per layer per batch
            self._subsample_stats(l, Hv.index_select(0, ctx.sub_idx).float())
        D, B, K, Kr = self.D, self.B, self.cfg.K, self.cfg.Kr
        cols = torch.arange(D, device=self.dev)
        for p, sgn in ((0, 1.0), (1, -1.0)):
            S = Hv if p == 0 else -Hv
            k = min(K, S.shape[0])
            v, i = torch.topk(S, k, dim=0)  # [k, D]
            g = ((l * 2 + p) * D + cols).expand(k, D).reshape(-1)
            v, i = v.float().reshape(-1), i.reshape(-1)
            ok = v > self.top_seed[g]
            self.stats["top_cand"] += int(ok.sum())
            grouped_merge(self.top_s, [self.top_t], g[ok], v[ok], [ctx.tok[i[ok]]], K)
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
            cap = self._cap(ctx)
            redo = [l for l, c in zip(filt, cnts) if c > cap]
            ok = [(l, c) for l, c in zip(filt, cnts) if c <= cap]
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
        top_thr = torch.maximum(self.top_s[:, K - 1], self.top_seed)
        res_thr = torch.maximum(self.res_s[:, Kr - 1], self.res_seed)
        top_g, top_v, top_t = [], [], []
        res_g, res_v, res_k, res_t = [], [], [], []
        for p, sgn in ((0, 1.0), (1, -1.0)):
            s = vals * sgn
            g = (lay * 2 + p) * D + cols
            m = s > top_thr[g]
            top_g.append(g[m]); top_v.append(s[m]); top_t.append(tok[m])
            if not self.cfg.reservoir:
                continue
            b = bin_of(s, self.E[lay, p, cols])
            gr = g * B + b.clamp(min=0)
            m = (b >= 0) & (key >= self.U_res) & (-key > res_thr[gr])
            res_g.append(gr[m]); res_k.append(key[m]); res_t.append(tok[m]); res_v.append(vals[m])
        # bulk bins: every element of the rows with key < U_res
        for l, _ in (ok if self.cfg.reservoir else []):
            R = self._pending[l][4]  # [n_r, D] float
            if R.shape[0] == 0:
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
        tg = torch.cat(top_g)
        self.stats["top_cand"] += tg.numel()
        grouped_merge(self.top_s, [self.top_t], tg, torch.cat(top_v), [torch.cat(top_t)], K)
        if not self.cfg.reservoir:
            return
        rg = torch.cat(res_g)
        self.stats["res_cand"] += rg.numel()
        grouped_merge(self.res_s, [self.res_t, self.res_v], rg, -torch.cat(res_k),
                      [torch.cat(res_t), torch.cat(res_v)], Kr)

    # ------------------------------------------------------------------ outputs
    def flush_gram(self):
        if self.gram is not None:
            self.gram64 += self.gram.double().cpu()
            self.gram.zero_()

    # ------------------------------------------------------------------ checkpoint / merge
    _STATE = ["top_s", "top_t", "res_s", "res_t", "res_v", "sub_sum", "sub_sq", "sub_bin", "E",
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
        v, i = torch.topk(torch.cat((self.top_s, other.top_s.to(self.dev)), 1), K, dim=1)
        self.top_t = torch.gather(torch.cat((self.top_t, other.top_t.to(self.dev)), 1), 1, i)
        self.top_s = v
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
            out["gram"] = self.gram64
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


class _tf32:
    """bf16-origin values are exact in TF32, so TF32 GEMMs give exact products with fp32 accumulation."""

    def __enter__(self):
        self.prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True

    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32 = self.prev
