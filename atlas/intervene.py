"""Post-GELU column interventions and their downstream effects in ESM-2."""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from .esm import MODELS

AA = "ACDEFGHIKLMNPQRSTVWY"


class Column(NamedTuple):
    layer: int
    r: Tensor


class Edit(NamedTuple):
    layer: int
    r: Tensor
    positions: list[int]
    values: list[float]


class Run(NamedTuple):
    lp: Tensor
    lp_full: Tensor
    post: dict[int, Tensor]
    x_final: Tensor


class Result(NamedTuple):
    dlp: Tensor
    later: list[tuple[int, int, float]]
    later_share: float | None
    write: Tensor


class Intervener:
    def __init__(self, model="8M", device="cpu", dtype=torch.float32):
        from transformers import EsmForMaskedLM, EsmTokenizer

        hf = MODELS.get(model, model)
        self.tok = EsmTokenizer.from_pretrained(hf)
        self.model = EsmForMaskedLM.from_pretrained(hf).to(device=device, dtype=dtype).eval()
        self.layers = self.model.esm.encoder.layer
        self.aa_ids = torch.tensor([self.tok.convert_tokens_to_ids(c) for c in AA], device=device)

    @torch.no_grad()
    def run(self, ids: Tensor, edit: Edit | None = None, capture=False) -> Run:
        post = {}
        final = []
        handles = []

        def hook(l):
            def apply(_m, _i, out):
                if edit is not None and edit[0] == l:
                    _, r, positions, values = edit
                    r = r.to(device=out.device, dtype=torch.float32)
                    p = torch.tensor(positions, device=out.device, dtype=torch.long) + 1
                    v = torch.tensor(values, device=out.device, dtype=torch.float32)
                    h = out[0, p].float()
                    a = h @ r
                    changed = v != a
                    if changed.any():
                        out = out.clone()
                        out[0, p[changed]] = (h[changed] + (v - a)[changed, None] * r[None, :]).to(out.dtype)
                if capture:
                    post[l] = out[0, 1:-1].clone()
                return out
            return apply

        def capture_final(_m, args):
            final.append(args[0][0, 1:-1].clone())

        try:
            for l, layer in enumerate(self.layers):
                handles.append(layer.intermediate.register_forward_hook(hook(l)))
            handles.append(self.model.esm.encoder.emb_layer_norm_after.register_forward_pre_hook(capture_final))
            ids = ids.to(self.model.device)
            lp_full = torch.log_softmax(self.model(input_ids=ids).logits[0].double(), -1)
        finally:
            for handle in handles:
                handle.remove()
        return Run(lp_full[1:-1, self.aa_ids], lp_full, post, final[0])

    @torch.no_grad()
    def replace(self, ids, column: Column, positions: list[int], values: list[float],
                k_later: int = 10, baseline: Run | None = None) -> Result:
        l, r = column
        if baseline is None:
            baseline = self.run(ids, capture=True)
        elif not baseline.post:
            baseline = baseline._replace(post=self.run(ids, capture=True).post)
        edited = self.run(ids, Edit(l, r, positions, values), capture=True)
        later = []
        later_share = None
        if l + 1 < len(self.layers):
            changes = torch.stack([(edited.post[j] - baseline.post[j]).abs().sum(0)
                                   for j in range(l + 1, len(self.layers))])
            flat = changes.flatten()
            order = torch.argsort(flat, descending=True, stable=True)[:k_later]
            width = changes.shape[1]
            later = [(l + 1 + i // width, i % width, float(flat[i])) for i in order.tolist()]
            total = float(flat.sum())
            later_share = float(flat[order].sum()) / total if total else 0.0
        a = self.column_values(baseline, column)[positions]
        v = torch.tensor(values, device=a.device, dtype=torch.float32)
        weight = self.layers[l].output.dense.weight.float()
        r = r.to(device=weight.device, dtype=torch.float32)
        write = (v - a).abs() * (weight @ r).norm()
        return Result(edited.lp - baseline.lp, later, later_share, write)

    @torch.no_grad()
    def head(self, x_final: Tensor) -> Tensor:
        x = x_final.to(device=self.model.device, dtype=self.model.dtype)
        logits = self.model.lm_head(self.model.esm.encoder.emb_layer_norm_after(x))
        return torch.log_softmax(logits.double(), -1)[:, self.aa_ids]

    @torch.no_grad()
    def column_values(self, run: Run, column: Column) -> Tensor:
        l, r = column
        h = run.post[l].float()
        return h @ r.to(device=h.device, dtype=torch.float32)
