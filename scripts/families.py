"""Measurement #4: four relations between the FFN units of a layer, and where they disagree.
Predictions: results/measurement_4_predictions.md.

  co-activation   Pearson r of pre-GELU h over the hashed subsample (atlas Gram, sums, squares)
  input weights   cos(W_in[i], W_in[j])        (intermediate.dense rows: what the units read)
  output weights  cos(W_out[:, i], W_out[:, j]) (output.dense columns: what the units write)
  tail overlap    Jaccard of the token sets above the --tail-bin edge (positive pole), from the bottom-k
                  reservoir sketches: one hash over all units, so the k smallest keys of the union of two
                  sketches are a uniform sample of the union, and the fraction of them in both estimates J.

Per layer: Spearman (average ranks) between relations over all pairs; top-5 neighbour agreement; counts of
near-duplicate tails; candidate disagreements (pairs extreme in one relation and unremarkable in another)
with their top-K residues, for writing up.

  python scripts/families.py --atlas slim.pt --data swissprot.pt --out dir [--model 8M]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.core import SEQ_STRIDE, token_keys  # noqa: E402
from atlas.esm import MODELS  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"


def avg_rank(x: torch.Tensor) -> torch.Tensor:
    o = torch.argsort(x, stable=True)
    xs = x[o]
    _, inv, cnt = torch.unique_consecutive(xs, return_inverse=True, return_counts=True)
    end = torch.cumsum(cnt, 0).double()
    avg = end - (cnt.double() - 1) / 2                              # 1-based average rank per tie group
    r = torch.empty_like(x, dtype=torch.float64)
    r[o] = avg[inv]
    return r


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra, rb = avg_rank(a.double()), avg_rank(b.double())
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / torch.sqrt((ra * ra).sum() * (rb * rb).sum()))


def tail_sketches(res_tok_l: torch.Tensor, bins: list[int], k: int, seed: int):
    """res_tok_l [D, B, Kr] (one layer, positive pole) -> per unit sorted bottom-k keys of the union of `bins`."""
    D = res_tok_l.shape[0]
    t = res_tok_l[:, bins].reshape(D, -1).long()
    key = torch.where(t >= 0, token_keys(t.clamp(min=0), seed), torch.full_like(t, 1 << 40))
    key, _ = torch.sort(key, dim=1)
    return key[:, :k]                                               # padded with 2^40 where fewer than k


def tail_jaccard(S: torch.Tensor, k: int) -> torch.Tensor:
    """All-pairs bottom-k Jaccard estimate [D, D] from sorted sketches S [D, k] (pad 2^40)."""
    D = S.shape[0]
    valid = S < (1 << 40)
    # pairs that share any key: sparse unit x key incidence, C = M M^T counts shared keys
    rows = torch.arange(D)[:, None].expand_as(S)[valid]
    _, col = torch.unique(S[valid], return_inverse=True)
    M = torch.sparse_coo_tensor(torch.stack([rows, col]), torch.ones(rows.numel()), (D, int(col.max()) + 1 if col.numel() else 1))
    C = torch.sparse.mm(M, M.t()).to_dense()
    J = torch.zeros(D, D)
    pairs = torch.nonzero(torch.triu(C, 1) > 0).tolist()
    for i, j in pairs:
        a, b = S[i][valid[i]], S[j][valid[j]]
        u = torch.unique(torch.cat([a, b]))[:k]                    # k smallest of the union (sorted)
        both = torch.isin(u, a) & torch.isin(u, b)
        J[i, j] = J[j, i] = both.float().mean()
    J.fill_diagonal_(1.0)
    return J


def top5(M: torch.Tensor) -> torch.Tensor:
    M = M.clone(); M.fill_diagonal_(-float("inf"))
    return torch.topk(M, 5, dim=1).indices


def agreement(A: torch.Tensor, B: torch.Tensor) -> float:
    ta, tb = top5(A), top5(B)
    ov = [(len(set(ta[i].tolist()) & set(tb[i].tolist())) / 5) for i in range(A.shape[0])]
    return float(torch.tensor(ov).median())


def residues(tok: torch.Tensor, seqs) -> str:
    out = []
    for t in tok.tolist():
        if t < 0:
            continue
        q, p = divmod(int(t), SEQ_STRIDE)
        out.append(seqs[q][p] if p < len(seqs[q]) else "?")
    c = {a: out.count(a) for a in set(out)}
    return " ".join(f"{a}{n}" for a, n in sorted(c.items(), key=lambda x: -x[1])[:5]) + f" ({len(set(int(t) // SEQ_STRIDE for t in tok.tolist() if t >= 0))} proteins)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="8M")
    ap.add_argument("--control", default="none", choices=["none", "shuffled"])
    ap.add_argument("--control-seed", type=int, default=0)
    ap.add_argument("--tail-bins", type=int, nargs="+", default=[3, 4, 5], help="bins whose union is the tail (3 = from q99.9)")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    at = torch.load(a.atlas, map_location="cpu", weights_only=False)
    from atlas.esm import load
    model, _, _ = load(a.model, a.control, dtype=torch.float32, device="cpu", seed=a.control_seed)
    seqs = torch.load(a.data, weights_only=False)["seq"]
    seed = int(at["cfg"]["seed"]) if isinstance(at.get("cfg"), dict) and "seed" in at["cfg"] else 0
    L = at["top_tok"].shape[0]; k = at["res_tok"].shape[-1]
    n = float(at["n_sub"]); assert int(at.get("gram_rows", at["n_sub"])) == int(at["n_sub"]), "Gram rows differ from the subsample"
    summ = {"atlas": a.atlas, "control": a.control, "control_seed": a.control_seed, "tail_bins": a.tail_bins, "k": k, "layers": {}}
    cases = []
    for l in (a.layers if a.layers is not None else range(L)):
        mu = at["sub_sum"][l].double() / n
        var = at["sub_sq"][l].double() / n - mu ** 2
        G = at["gram"][l].double() / n - mu[:, None] * mu[None, :]
        sd = torch.sqrt(var.clamp(min=1e-12))
        R1 = (G / (sd[:, None] * sd[None, :])).float()
        W_in = model.encoder.layer[l].intermediate.dense.weight.float()
        W_out = model.encoder.layer[l].output.dense.weight.float()
        Wi = torch.nn.functional.normalize(W_in, dim=1); Wo = torch.nn.functional.normalize(W_out.T, dim=1)
        R2, R3 = Wi @ Wi.T, Wo @ Wo.T
        S = tail_sketches(at["res_tok"][l, 0], a.tail_bins, k, seed)
        R4 = tail_jaccard(S, k)
        D = R1.shape[0]
        iu = torch.triu_indices(D, D, 1)
        rel = {"coact": R1, "input": R2, "output": R3, "tail": R4}
        v = {name: M[iu[0], iu[1]] for name, M in rel.items()}
        names = list(rel)
        sp = {f"{x}~{y}": spearman(v[x], v[y]) for i, x in enumerate(names) for y in names[i + 1:]}
        ag = {f"{x}~{y}": agreement(rel[x], rel[y]) for i, x in enumerate(names) for y in names[i + 1:]}
        tj = v["tail"]
        summ["layers"][str(l)] = {"spearman": sp, "top5_agreement_median": ag,
                                  "pairs": int(tj.numel()), "pairs_tail_ge_0.5": int((tj >= 0.5).sum()),
                                  "pairs_tail_gt_0": int((tj > 0).sum()),
                                  "coact_quantiles": [round(float(x), 3) for x in torch.quantile(v["coact"][:200000], torch.tensor([0.01, 0.5, 0.99]))]}
        # disagreement candidates: (i) same tail, different writes; (ii) high co-activation, disjoint tails;
        # (iii) near-identical input weights, different tails
        def pick(mask, score, kind, m=3):
            idx = torch.nonzero(mask).flatten()
            if idx.numel() == 0:
                return
            idx = idx[torch.argsort(score[idx], descending=True)[:m]]
            for t in idx.tolist():
                i, j = int(iu[0, t]), int(iu[1, t])
                cases.append({"kind": kind, "layer": l, "i": i, "j": j, **{name: round(float(v[name][t]), 3) for name in names},
                              "top_i": residues(at["top_tok"][l, 0, i], seqs), "top_j": residues(at["top_tok"][l, 0, j], seqs)})
        pick((v["tail"] >= 0.5) & (v["output"] <= 0.0), v["tail"] - v["output"], "same tail, different writes")
        pick((v["coact"] >= 0.7) & (v["tail"] == 0), v["coact"], "high co-activation, disjoint tails")
        pick((v["input"] >= 0.8) & (v["tail"] <= 0.05), v["input"], "near-identical input weights, different tails")
        print(json.dumps({"layer": l, **summ["layers"][str(l)]}), flush=True)
    (out / "families.json").write_text(json.dumps(summ, indent=1))
    (out / "disagreements.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n")


if __name__ == "__main__":
    main()
