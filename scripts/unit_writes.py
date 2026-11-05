"""Measurement #3: what a unit writes, and what silencing it does, at its positive extreme, its median token and
its negative extreme (ESM-2, fp32, CPU). Predictions: results/measurement_3_predictions.md.

Per (unit, level): h (pre-GELU), GELU(h), written norm |GELU(h)| * ||W_out[:, u]|| against the residual it is
added to (EsmOutput adds the FFN write to the attention output), and silencing: the unit's GELU output set to 0
at that one position; effect on the logits at that position (delta log p of the true residue, KL(p || p_sil))
and the largest KL at any other position ("travel"). Baseline: a random other unit of the same layer silenced
at the same position.

  python scripts/unit_writes.py --data data/swissprot_2026_03.pt --exclude-acc lists/heldout.tsv --out dir
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.esm import MODELS  # noqa: E402

LEVELS = ("positive extreme", "median token", "negative extreme")


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 2 ** 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8M")
    ap.add_argument("--data", required=True)
    ap.add_argument("--exclude-acc", default=None)
    ap.add_argument("--proteins", type=int, default=400)
    ap.add_argument("--units", type=int, default=50, help="random units per layer")
    ap.add_argument("--max-res", type=int, default=1022)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    from transformers import EsmForMaskedLM, EsmTokenizer
    hf = MODELS.get(a.model, a.model)
    tok = EsmTokenizer.from_pretrained(hf)
    model = EsmForMaskedLM.from_pretrained(hf).float().eval()
    layers = model.esm.encoder.layer
    L, D = len(layers), model.config.intermediate_size

    d = torch.load(a.data, weights_only=False)
    accs, seqs = d["acc"], d["seq"]
    ok = list(range(len(accs)))
    if a.exclude_acc:
        ex = {ln.split("\t")[0].split(",")[0].strip() for ln in open(a.exclude_acc) if ln.strip() and not ln.startswith("Entry")}
        ok = [i for i in ok if accs[i] not in ex]
    g = torch.Generator().manual_seed(a.seed)
    pick = [ok[i] for i in torch.randperm(len(ok), generator=g)[: a.proteins].tolist()]
    units = [torch.randperm(D, generator=g)[: a.units] for _ in range(L)]

    # pass 1: h for the chosen units, residual norms, log-probs
    cap = {}
    hooks = []
    for l, lay in enumerate(layers):
        hooks.append(lay.intermediate.dense.register_forward_hook(lambda _m, _i, o, l=l: cap.__setitem__(("h", l), o[0])))
        hooks.append(lay.output.register_forward_pre_hook(
            lambda _m, args, kw, l=l: cap.__setitem__(("res", l), (args[1] if len(args) > 1 else kw["input_tensor"])[0].norm(dim=-1)),
            with_kwargs=True))
    ids, H, RES, LP = [], [[] for _ in range(L)], [[] for _ in range(L)], []
    for i in pick:
        x = tok(seqs[i][: a.max_res], return_tensors="pt")["input_ids"]
        lp = torch.log_softmax(model(input_ids=x).logits[0].double(), -1)
        ids.append(x); LP.append(lp)
        for l in range(L):
            H[l].append(cap[("h", l)][1:-1, units[l]].clone())       # residues only (no BOS/EOS)
            RES[l].append(cap[("res", l)][1:-1].clone())
    for h in hooks:
        h.remove()
    n_res = sum(x.shape[1] - 2 for x in ids)
    owner = torch.cat([torch.full((x.shape[1] - 2,), j) for j, x in enumerate(ids)])
    pos = torch.cat([torch.arange(x.shape[1] - 2) for x in ids])

    # silencing: one unit's GELU output zeroed at one position
    def silenced(j, l, u, p):
        def hk(_m, _i, o):
            o = o.clone(); o[0, p + 1, u] = 0.0
            return o
        hd = layers[l].intermediate.register_forward_hook(hk)
        try:
            return torch.log_softmax(model(input_ids=ids[j]).logits[0].double(), -1)
        finally:
            hd.remove()

    def effect(j, l, u, p):
        lp0 = LP[j]; lp1 = silenced(j, l, u, p)
        t = int(ids[j][0, p + 1])
        kl = (lp0.exp() * (lp0 - lp1)).sum(-1).clamp(min=0)            # [L+2], fp64 from fp32 logits
        other = torch.cat([kl[1:p + 1], kl[p + 2:-1]])
        return {"dlogp_true": float(lp1[p + 1, t] - lp0[p + 1, t]), "kl": float(kl[p + 1]),
                "travel_kl": float(other.max()) if other.numel() else 0.0}

    # noise floor: the same forward again, nothing silenced (CPU thread scheduling can change float order)
    floor = max(float((torch.log_softmax(model(input_ids=ids[j]).logits[0].double(), -1) - LP[j]).abs().max()) for j in range(min(5, len(ids))))
    rows = []
    for l in range(L):
        Hl = torch.cat(H[l]); Rl = torch.cat(RES[l])
        Wn = layers[l].output.dense.weight.norm(dim=0)                 # [D] ||W_out[:, u]||
        med = Hl.median(0).values
        for c, u in enumerate(units[l].tolist()):
            col = Hl[:, c]
            for lev, r in zip(LEVELS, (int(col.argmax()), int((col - med[c]).abs().argmin()), int(col.argmin()))):
                j, p = int(owner[r]), int(pos[r])
                hv = float(col[r]); gv = float(gelu(torch.tensor(hv)))
                v = u
                while v == u:
                    v = int(torch.randint(D, (1,), generator=g))
                e, b = effect(j, l, u, p), effect(j, l, v, p)
                rows.append({"layer": l, "unit": u, "level": lev, "acc": accs[pick[j]], "pos": p, "h": hv, "gelu": gv,
                             "written": abs(gv) * float(Wn[u]), "residual": float(Rl[r]), **e,
                             "base_unit": v, **{f"base_{k}": x for k, x in b.items()}})
    (out / "cases.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def med_(xs):
        return statistics.median(xs) if xs else None
    summ = {}
    for l in range(L):
        for lev in LEVELS:
            R = [r for r in rows if r["layer"] == l and r["level"] == lev]
            summ[f"layer {l} / {lev}"] = {
                "n": len(R), "h": med_([r["h"] for r in R]), "gelu": med_([r["gelu"] for r in R]),
                "abs_gelu": med_([abs(r["gelu"]) for r in R]),
                "written_over_residual": med_([r["written"] / r["residual"] for r in R]),
                "kl": med_([r["kl"] for r in R]), "abs_dlogp_true": med_([abs(r["dlogp_true"]) for r in R]),
                "base_kl": med_([r["base_kl"] for r in R]),
                "kl_over_base_medians": med_([r["kl"] for r in R]) / max(med_([r["base_kl"] for r in R]), 1e-15),
                "travel_kl": med_([r["travel_kl"] for r in R]), "max_travel_kl": max(r["travel_kl"] for r in R),
                "frac_abs_dlogp_gt_0.1": sum(abs(r["dlogp_true"]) > 0.1 for r in R) / max(len(R), 1)}
    res = {"model": hf, "proteins": len(pick), "residues": n_res, "units_per_layer": a.units, "seconds": round(time.time() - t0, 1),
           "noise_floor_max_abs_dlogp": floor, "threads": torch.get_num_threads(),
           "summary": summ}
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"proteins": len(pick), "residues": n_res, "cases": len(rows), "seconds": res["seconds"], "noise_floor": floor}))


if __name__ == "__main__":
    main()
