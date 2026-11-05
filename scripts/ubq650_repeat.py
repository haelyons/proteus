"""Hypothesis D for 650M layer 15 unit 2087 (results/ubq650_unit_predictions.md): does the extreme response at
ubiquitin H68 need tandem copies in the same protein? Also checks whether the unit's non-ubiquitin peak samples
come from proteins that repeat their window."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.esm import load  # noqa: E402

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def main():
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    model, tok, _ = load("650M", "none", dtype=torch.float32, device="cpu")
    cap = {}
    model.encoder.layer[15].intermediate.dense.register_forward_hook(lambda _m, _i, o: cap.__setitem__("h", o[0, :, 2087]))

    def acts(s):
        model(**tok(s[:1022], return_tensors="pt"))
        return cap["h"][1:-1].clone()
    d = torch.load(sys.argv[1], weights_only=False)
    idx = {x: i for i, x in enumerate(d["acc"])}
    res = {}
    for acc, name in (("P0CG48", "UBC (9 copies)"), ("P0CG47", "UBB (3 copies)"), ("P62987", "UBA52 (1 copy + L40)"), ("P62979", "RPS27A (1 copy + S27a)")):
        s = d["seq"][idx[acc]]; v = acts(s)
        starts = [i for i in range(len(s) - 75) if s[i:i + 76] == UBQ]
        res[name] = [round(float(v[st + 67]), 3) for st in starts]
    rng = random.Random(1)
    spacer = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(40))
    for name, s, h in (("Ub+Ub", UBQ + UBQ, [67, 76 + 67]), ("Ub+40+Ub", UBQ + spacer + UBQ, [67, 116 + 67])):
        v = acts(s); res[name] = [round(float(v[i]), 3) for i in h]
    sumo = d["seq"][idx["P63165"]][:97]
    v1 = acts(sumo); v2 = acts(sumo + sumo)
    res["SUMO1 x2: max over copy 1 / copy 2 / single SUMO1"] = [round(float(v2[:97].max()), 3), round(float(v2[97:].max()), 3), round(float(v1.max()), 3)]
    other = d["seq"][idx["P00698"]][18:98]                                   # hen lysozyme fragment, unrelated
    v1 = acts(other); v2 = acts(other + other)
    res["unrelated 80-mer x2 (lysozyme 19-98): max copy 1 / copy 2 / single"] = [round(float(v2[:80].max()), 3), round(float(v2[80:].max()), 3), round(float(v1.max()), 3)]
    # peak-bin samples outside ubiquitin: does the protein repeat the sample's +-5 window elsewhere?
    U = json.load(open("infra/runs/atlas-ru-20261105-1704/ubq650_units.json"))
    sel = next(o for o in U if o["layer"] == 15 and o["unit"] == 2087)
    rep, tot, ex = 0, 0, []
    for t in sel["res_tok"][5]:
        if t < 0:
            continue
        si, p = divmod(t, 1024); s = d["seq"][si]
        w = s[max(0, p - 5):p + 6]
        if w in UBQ:
            continue
        tot += 1
        n = s.count(w)
        rep += n > 1
        if len(ex) < 6:
            ex.append((d["acc"][si], p + 1, s[p], w, n))
    res["peak-bin samples outside ubiquitin: window occurs more than once in its protein"] = f"{rep} / {tot}"
    res["examples (acc, pos, residue, window, occurrences)"] = ex
    (out / "repeat.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
