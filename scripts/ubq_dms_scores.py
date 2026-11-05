"""Score every single substitution of the scan's wild type (yeast ubiquitin) with ESM-2 masked marginals and
compare with the deep mutational scan per condition, DMSO primary (predictions:
results/mechanistic_round1_predictions.md section 3). Reads the scan's per-condition CSVs, never its pickles.

  score(pos, mut) = log p(mut | sequence with pos masked) - log p(wt | same)

  python scripts/ubq_dms_scores.py --dir path/to/pubs2014 --out dir [--models 8M 650M]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.esm import MODELS  # noqa: E402
from scripts.dms_align import CONDS, read  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"


def spearman(x, y):
    x, y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    rx = torch.argsort(torch.argsort(x)).double(); ry = torch.argsort(torch.argsort(y)).double()
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / torch.sqrt((rx * rx).sum() * (ry * ry).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=["8M", "650M"])
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    D = {c: read(Path(a.dir) / f) for c, f in CONDS.items()}
    wt = {}
    for (pos, w, _m) in D["DMSO"]:
        wt[pos] = w
    yeast = "".join(wt[p] for p in sorted(wt))
    from transformers import EsmForMaskedLM, EsmTokenizer
    res = {"wild_type": yeast, "models": {}}
    for mname in a.models:
        hf = MODELS[mname]
        tok = EsmTokenizer.from_pretrained(hf); model = EsmForMaskedLM.from_pretrained(hf).float().eval()
        ids = tok(yeast, return_tensors="pt")["input_ids"]
        aa_ids = torch.tensor([tok.convert_tokens_to_ids(c) for c in AA])
        score = {}
        for i in range(len(yeast)):
            x = ids.clone(); x[0, i + 1] = tok.mask_token_id
            lp = torch.log_softmax(model(input_ids=x).logits[0, i + 1].double(), -1)[aa_ids]
            w = AA.index(yeast[i])
            for k, m in enumerate(AA):
                if m != yeast[i]:
                    score[(i + 1, yeast[i], m)] = float(lp[k] - lp[w])
        per = {}
        for c in D:
            ks = [k for k in score if D[c].get(k) is not None]
            per[c] = {"n": len(ks), "spearman": round(spearman([score[k] for k in ks], [D[c][k] for k in ks]), 3)}
        res["models"][mname] = {"spearman_by_condition": per, "primary_DMSO": per["DMSO"]}
        (out / f"scores_{mname}.json").write_text(json.dumps({f"{w}{p}{m}": round(v, 4) for (p, w, m), v in score.items()}))
        del model
        print(mname, json.dumps(per), flush=True)
    (out / "ubq_dms_scores.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
