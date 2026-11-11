"""Swiss-Prot batching: seeded shuffle, length bucketing within chunks, truncation to
max_res residues, packed token ids, per-residue annotation ids from intervals."""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import torch

from .core import pack_tok

AA = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class Batch:
    input_ids: torch.Tensor      # [B, Lmax] int64
    attention_mask: torch.Tensor  # [B, Lmax] int64
    valid_rows: torch.Tensor     # [n_v] positions in the flattened [B*Lmax] rows
    tok: torch.Tensor            # [n_v] packed (seq_idx, pos)
    ann: torch.Tensor            # [n_v, M_a] annotation ids, -1 padded
    seq_idx: torch.Tensor        # [B]
    n_rows: int                  # B * Lmax (rows seen by hooks, incl. padding)


class SwissProt:
    def __init__(self, path: str, max_res: int = 1022, max_ann: int = 8):
        d = torch.load(path, weights_only=False)
        self.d = d
        self.seq = d["seq"]
        self.len = d["len"].long()
        self.vocab = d["vocab"]
        self.max_res, self.max_ann = max_res, max_ann
        iv = d["iv"].long()
        iv = iv[torch.argsort(iv[:, 0], stable=True)]
        self.iv = iv
        cnt = torch.bincount(iv[:, 0], minlength=len(self.seq))
        self.iv_ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(cnt, 0)])
        self.ann_dropped = 0

    def __len__(self):
        return len(self.seq)

    def coverage(self, idx: torch.Tensor | None = None) -> dict:
        L = self.len if idx is None else self.len[idx]
        drop = (L - self.max_res).clamp(min=0)
        return {"n_seq": int(L.numel()), "n_res_full": int(L.sum()), "n_res_used": int(L.clamp(max=self.max_res).sum()),
                "n_truncated": int((drop > 0).sum()), "frac_seq_truncated": float((drop > 0).float().mean()),
                "frac_res_dropped": float(drop.sum() / L.sum())}

    def plan(self, order: torch.Tensor, tokens_per_batch: int, window: int = 8192, seed: int = 0,
             shuffle_batches: bool = True) -> list[list[int]]:
        """Deterministic batch plan (lists of seq indices). Sequences are length-bucketed only
        within consecutive windows of `window` sequences of the (shuffled) order; batches are
        shuffled within each window unless shuffle_batches=False (then ascending length)."""
        g = torch.Generator().manual_seed(seed)
        out = []
        for c0 in range(0, len(order), window):
            ch = order[c0:c0 + window]
            L = self.len[ch].clamp(max=self.max_res)
            ch = ch[torch.argsort(L, stable=True)]
            groups, cur, cur_max = [], [], 0
            for i in ch.tolist():
                n = min(int(self.len[i]), self.max_res) + 2
                m = max(cur_max, n)
                if cur and m * (len(cur) + 1) > tokens_per_batch:
                    groups.append(cur)
                    cur, m = [], n
                cur.append(i)
                cur_max = m
            if cur:
                groups.append(cur)
            perm = torch.randperm(len(groups), generator=g).tolist() if shuffle_batches else range(len(groups))
            out.extend(groups[gi] for gi in perm)
        return out

    def batches(self, tokenizer_vocab: dict, order: torch.Tensor, tokens_per_batch: int,
                chunk: int = 8192, seed: int = 0, start: int = 0, plan: list | None = None,
                shuffle_batches: bool = True):
        """Yield Batch objects from batch index `start` of the deterministic plan."""
        plan = plan if plan is not None else self.plan(order, tokens_per_batch, chunk, seed, shuffle_batches)
        aa_tok = {c: tokenizer_vocab.get(c, tokenizer_vocab["<unk>"]) for c in set(AA) | set("XBUZO")}
        cls_, eos, pad = tokenizer_vocab["<cls>"], tokenizer_vocab["<eos>"], tokenizer_vocab["<pad>"]
        for grp in plan[start:]:
            yield self._make(grp, aa_tok, cls_, eos, pad)

    def pilot_plan(self, plan: list[list[int]], pilot_tokens: int, tokens_per_batch: int, seed: int = 0) -> list[list[int]]:
        """Pilot batches from sequences drawn uniformly at random from the whole plan (all
        lengths), rather than the plan's first, length-homogeneous batches. Every pilot
        sequence is also streamed later, which is what makes the top-K seed exact."""
        seqs = [i for grp in plan for i in grp]
        perm = torch.randperm(len(seqs), generator=torch.Generator().manual_seed(seed + 7)).tolist()
        pick, n = [], 0
        for j in perm:
            i = seqs[j]
            pick.append(i)
            n += min(int(self.len[i]), self.max_res)
            if n >= pilot_tokens:
                break
        return self.plan(torch.tensor(pick), tokens_per_batch, window=len(pick), seed=seed)

    def _make(self, idx, aa_tok, cls_, eos, pad) -> Batch:
        B = len(idx)
        lens = [min(int(self.len[i]), self.max_res) for i in idx]
        Lmax = max(lens) + 2
        ids = torch.full((B, Lmax), pad, dtype=torch.long)
        am = torch.zeros((B, Lmax), dtype=torch.long)
        rows, toks, anns = [], [], []
        for r, (i, n) in enumerate(zip(idx, lens)):
            s = self.seq[i][:n]
            ids[r, 0] = cls_
            ids[r, 1:n + 1] = torch.tensor([aa_tok.get(c, aa_tok["X"]) for c in s])
            ids[r, n + 1] = eos
            am[r, :n + 2] = 1
            rows.append(r * Lmax + 1 + torch.arange(n))
            toks.append(pack_tok(torch.full((n,), i), torch.arange(n)))
            anns.append(self._ann(i, s, n))
        return Batch(ids, am, torch.cat(rows), torch.cat(toks), torch.cat(anns),
                     torch.tensor(idx), B * Lmax)

    def _ann(self, i, s, n) -> torch.Tensor:
        A = torch.full((n, self.max_ann), -1, dtype=torch.long)
        A[:, 0] = torch.tensor([AA.index(c) if c in AA else 20 for c in s])
        iv = self.iv[self.iv_ptr[i]:self.iv_ptr[i + 1]]
        fill = torch.ones(n, dtype=torch.long)
        for _, a0, a1, k in iv.tolist():
            a1 = min(a1, n)
            if a0 >= a1:
                continue
            pos = torch.arange(a0, a1)
            ok = fill[pos] < self.max_ann
            self.ann_dropped += int((~ok).sum())
            pos = pos[ok]
            A[pos, fill[pos]] = k
            fill[pos] += 1
        return A


def shuffled_order(n: int, seed: int = 0) -> torch.Tensor:
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed))


def plan_digest(plan) -> str:
    """SHA256 of the length-prefixed global plan as little-endian int64."""
    digest = hashlib.sha256(struct.pack("<q", len(plan)))
    for group in plan:
        digest.update(struct.pack("<q", len(group)))
        digest.update(struct.pack(f"<{len(group)}q", *group))
    return digest.hexdigest()
