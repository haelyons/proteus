"""Probes for 650M layer 15 unit 2087 (predictions: results/ubq650_unit_predictions.md). Pre-GELU value of the unit
at named residues of designed sequences; ESM-2 650M, fp32, CPU.

  python scripts/ubq650_probe.py --data swissprot.pt --out dir
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.esm import load  # noqa: E402

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
H68 = 67                                                     # 0-based


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--unit", type=int, default=2087)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    model, tok, _ = load("650M", "none", dtype=torch.float32, device="cpu")
    cap = {}
    model.encoder.layer[a.layer].intermediate.dense.register_forward_hook(lambda _m, _i, o: cap.__setitem__("h", o[0, :, a.unit]))

    def acts(s):
        model(**tok(s, return_tensors="pt"))
        return cap["h"][1:-1].clone()
    d = torch.load(a.data, weights_only=False)
    idx = {x: i for i, x in enumerate(d["acc"])}
    seq = lambda acc: d["seq"][idx[acc]]  # noqa: E731
    rng = random.Random(0)
    res = {}
    base = float(acts(UBQ)[H68]); res["ubiquitin H68"] = base
    full = acts(UBQ)
    res["ubiquitin: top 3 positions (1-based, value)"] = [(int(i) + 1, UBQ[int(i)], round(float(full[i]), 3)) for i in torch.topk(full, 3).indices]
    nedd8 = seq("Q15843")[:76]; hn = nedd8.find("LHLVL") + 1
    res["NEDD8 H (LHLVL)"] = float(acts(nedd8)[hn])
    # transplant ubiquitin 61-75 into unrelated proteins
    ubq_accs = {l.split("\t")[0] for l in open("results/batch_20261105/ubiquitin_copies.tsv")}
    pool = [i for i in range(len(d["acc"])) if 150 <= len(d["seq"][i]) <= 400 and d["acc"][i] not in ubq_accs]
    win = UBQ[60:75]; trans = []
    for i in rng.sample(pool, 10):
        s = d["seq"][i]; p = rng.randint(20, len(s) - 35)
        new = s[:p] + win + s[p + 15:]
        trans.append(float(acts(new)[p + 7]))
    res["transplant 61-75 into 10 proteins, at H (values)"] = [round(v, 3) for v in trans]
    res["transplant median / isolated"] = st.median(trans) / base
    # distal shuffle 1-50
    sh = []
    for _ in range(10):
        head = list(UBQ[:50]); rng.shuffle(head)
        sh.append(float(acts("".join(head) + UBQ[50:])[H68]))
    res["residues 1-50 shuffled (10), at H68"] = [round(v, 3) for v in sh]
    res["distal shuffle median / isolated"] = st.median(sh) / base
    # point mutants
    for name, pos, aa in (("L8A", 7, "A"), ("I44A", 43, "A"), ("V70A", 69, "A"), ("L67A", 66, "A"), ("L69A", 68, "A"), ("H68A", 67, "A")):
        m = UBQ[:pos] + aa + UBQ[pos + 1:]
        v = acts(m)
        res[f"mutant {name}, at 68"] = float(v[H68])
    # family members and beta-grasp folds
    park = seq("O60260")[:76]; hp = park.find("VHIV") + 1
    res["Parkin UBL H (VHIV)"] = float(acts(park)[hp])
    for acc, name, L in (("P63165", "SUMO1", 97), ("P54725", "RAD23A UBL", 80), ("P05161", "ISG15", 165)):
        s = seq(acc)[:L]; v = acts(s)
        hs = [i for i, c in enumerate(s) if c == "H"]
        res[f"{name}: max over sequence (pos, residue, value)"] = (int(torch.argmax(v)) + 1, s[int(torch.argmax(v))], round(float(v.max()), 3))
        res[f"{name}: max at any His"] = round(max(float(v[i]) for i in hs), 3) if hs else None
    # composition-preserving shuffles of ubiquitin
    shuf = []
    for _ in range(20):
        s = list(UBQ); rng.shuffle(s); s = "".join(s)
        v = acts(s); hs = [i for i, c in enumerate(s) if c == "H"]
        shuf.append(max(float(v[i]) for i in hs))
    res["ubiquitin shuffles (20): max at His"] = [round(v, 3) for v in shuf]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "probes.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
