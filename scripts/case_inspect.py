"""Discovery-only inspection of one 8M unit (no fresh proteins): where its top activations sit.

On the discovery proteins (non-held-out, seed 2, the same 600 as `case_units.py discover`):
  - all tokens above the unit's q99.9 (no per-protein cap, no margin): share within 3 residues of either terminus,
    relative position (deciles), proteins contributing;
  - residue log2 enrichment at -3..+3 against all tokens, and state5 / burial3 shares against base;
and on the atlas reservoirs: share of sampled tokens within 3 residues of a terminus, per bin.

  python scripts/case_inspect.py --unit 3,1238 --data .. --desc-dir .. --uniref .. --atlas .. --out dir
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.core import SEQ_STRIDE  # noqa: E402
from scripts.case_units import Model  # noqa: E402
from scripts.pair_balance import burial3_of, proteins  # noqa: E402
from scripts.synthetic_columns import state5  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", required=True); ap.add_argument("--data", required=True); ap.add_argument("--desc-dir", required=True)
    ap.add_argument("--uniref", required=True); ap.add_argument("--atlas", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600); ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    l, u = map(int, a.unit.split(","))
    d = torch.load(a.data, weights_only=False)
    at = torch.load(a.atlas, map_location="cpu", weights_only=False)
    term = {}
    for b, name in ((5, "peak"), (4, "q99.99-q99.999"), (3, "q99.9-q99.99"), (2, "q99-q99.9"), (1, "q90-q99"), (0, "q50-q90")):
        n = t = nt = ct = 0
        for k in at["res_tok"][l, 0, u, b].tolist():
            if k < 0:
                continue
            si, pos = divmod(int(k), SEQ_STRIDE); L = len(d["seq"][si])
            if pos >= L:
                continue
            n += 1; nt += pos < 3; ct += pos >= L - 3
        term[name] = {"n": n, "N-term (pos < 3)": round(nt / max(n, 1), 3), "C-term (last 3)": round(ct / max(n, 1), 3)}
    M = Model()
    prot = proteins(a, False, a.n, a.seed)
    b3, _ = burial3_of(prot)
    H = []
    for p in prot:
        M.run(p["seq"]); H.append(M.cap[("h", l)][1:-1, u].clone())
    allh = torch.cat(H)
    q999 = float(torch.quantile(allh, 0.999))
    top = [(j, i) for j, h in enumerate(H) for i in torch.nonzero(h >= q999).flatten().tolist()]
    L = lambda j: len(prot[j]["seq"])  # noqa: E731
    nterm = sum(i < 3 for j, i in top); cterm = sum(i >= L(j) - 3 for j, i in top)
    dec = Counter(min(int(10 * i / L(j)), 9) for j, i in top)
    interior = [(j, i) for j, i in top if 3 <= i < L(j) - 3]
    c, cb = Counter(), Counter(); s5, s5b, bb, bbb = Counter(), Counter(), Counter(), Counter(); n = nb = 0
    for j, p in enumerate(prot):
        st = state5(p)
        for i in range(len(p["seq"])):
            nb += 1; s5b[int(st[i])] += 1; bbb[int(b3[j][i])] += 1
            for o in range(-3, 4):
                if 0 <= i + o < len(p["seq"]):
                    cb[(o, p["seq"][i + o])] += 1
    for j, i in interior:
        p = prot[j]; n += 1; s5[int(state5(p)[i])] += 1; bb[int(b3[j][i])] += 1
        for o in range(-3, 4):
            c[(o, p["seq"][i + o])] += 1
    enr = sorted(((round(math.log2(((c[k] + 1) / (n + 20)) / ((cb[k] + 1) / (nb + 20))), 2), k[0], k[1], c[k]) for k in cb if c[k] >= 3), reverse=True)
    res = {"unit": [l, u], "atlas_terminus_share_by_bin": term, "discovery": {
        "proteins": len(prot), "tokens": int(allh.shape[0]), "q99.9": q999, "q99": float(torch.quantile(allh, 0.99)),
        "above_q99.9": len(top), "proteins_above": len({j for j, _ in top}),
        "N-term share (pos < 3)": round(nterm / max(len(top), 1), 3), "C-term share (last 3)": round(cterm / max(len(top), 1), 3),
        "relative position deciles": [dec[k] for k in range(10)], "interior": n,
        "interior +-3 enriched (log2, offset, residue, count)": enr[:15], "interior +-3 depleted": enr[-6:],
        "interior state5 (0 loop 1 helix 2 strand 3 disordered 4 uncertain)": {k: round(s5[k] / max(n, 1), 3) for k in range(5)},
        "base state5": {k: round(s5b[k] / nb, 3) for k in range(5)},
        "interior burial3 (-1 no pLDDT>=70, 0 exposed .. 2 buried)": {k: round(bb[k] / max(n, 1), 3) for k in (-1, 0, 1, 2)},
        "base burial3": {k: round(bbb[k] / nb, 3) for k in (-1, 0, 1, 2)}}}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"inspect_{l}_{u}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
