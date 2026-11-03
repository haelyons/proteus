"""ESM-2 (HuggingFace) loading, controls, and hooks feeding the Atlas.

Hook point: encoder.layer[i].intermediate.dense output = pre-GELU FFN activation [B, L, d_ff].
Controls:
  shuffled : values permuted within every weight and bias tensor (InterPLM's control:
             "we shuffled the values within each weight and bias of ESM2-8M"); seeded.
  random   : fresh random init from the config (HF _init_weights); seeded.
"""
from __future__ import annotations

import torch

MODELS = {
    "8M": "facebook/esm2_t6_8M_UR50D",
    "35M": "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
}


def load(name: str, control: str = "none", dtype=torch.bfloat16, device="cpu", seed: int = 0):
    from transformers import EsmConfig, EsmModel, EsmTokenizer

    hf = MODELS.get(name, name)
    tok = EsmTokenizer.from_pretrained(hf)
    kw = {}
    try:
        kw["attn_implementation"] = "sdpa"
        model = EsmModel.from_pretrained(hf, add_pooling_layer=False, **kw)
    except (ValueError, TypeError):
        kw = {}
        model = EsmModel.from_pretrained(hf, add_pooling_layer=False)
    if control == "shuffled":
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for _, p in model.named_parameters():
                flat = p.data.view(-1)
                flat.copy_(flat[torch.randperm(flat.numel(), generator=g)])
    elif control == "random":
        torch.manual_seed(seed)
        model = EsmModel(EsmConfig.from_pretrained(hf), add_pooling_layer=False)
    elif control != "none":
        raise ValueError(control)
    model = model.to(device=device, dtype=dtype).eval()
    info = {"hf": hf, "control": control, "attn": getattr(model.config, "_attn_implementation", "eager"),
            "n_layers": model.config.num_hidden_layers, "d_model": model.config.hidden_size,
            "d_ff": model.config.intermediate_size,
            "n_params": sum(p.numel() for p in model.parameters())}
    return model, tok, info


def ffn_modules(model):
    return [layer.intermediate.dense for layer in model.encoder.layer]


def attach(model, sink):
    """sink(l, H[B*L, d_ff]) is called from each FFN pre-activation hook. Returns handles."""
    handles = []
    for i, mod in enumerate(ffn_modules(model)):
        def hook(_m, _inp, out, i=i):
            sink(i, out.reshape(-1, out.shape[-1]))
        handles.append(mod.register_forward_hook(hook))
    return handles


def flops_per_batch(info: dict, n_rows: int, Lmax: int) -> float:
    """Dense matmul FLOPs for one forward over n_rows padded positions (2 per MAC):
    per layer 2*(4 d^2 + 2 d d_ff) per row for linears + 4*Lmax*d per row for attention."""
    d, f, L = info["d_model"], info["d_ff"], info["n_layers"]
    return n_rows * L * (2 * (4 * d * d + 2 * d * f) + 4 * Lmax * d)


@torch.no_grad()
def pilot(model, batches, device, max_tokens: int, store_device="cpu"):
    """Collect valid-token FFN activations per layer from the first batches -> list[[N, d_ff]]."""
    L = model.config.num_hidden_layers
    bufs = [[] for _ in range(L)]
    cur = {}

    def sink(l, H):
        bufs[l].append(H.index_select(0, cur["rows"]).to(store_device))

    hs = attach(model, sink)
    n = 0
    try:
        for b in batches:
            cur["rows"] = b.valid_rows.to(device)
            model(input_ids=b.input_ids.to(device), attention_mask=b.attention_mask.to(device))
            n += b.valid_rows.numel()
            if n >= max_tokens:
                break
    finally:
        for h in hs:
            h.remove()
    return [torch.cat(x) for x in bufs], n
