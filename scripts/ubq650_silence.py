"""Downstream test for 650M layer 15 unit 2087 (results/ubq650_unit_predictions.md, last section)."""
import json, random, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import EsmForMaskedLM, EsmTokenizer  # noqa: E402
UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
torch.set_grad_enabled(False)
hf = "facebook/esm2_t33_650M_UR50D"; tok = EsmTokenizer.from_pretrained(hf); m = EsmForMaskedLM.from_pretrained(hf).float().eval()
lay = m.esm.encoder.layer[15]; wn = lay.output.dense.weight.norm(dim=0)
st = {"zero": None, "post": None}
def hook(_m, _i, o):
    st["post"] = o[0].clone()
    if st["zero"] is not None:
        o = o.clone(); o[0, :, st["zero"]] = 0.0
    return o
lay.intermediate.register_forward_hook(hook)
def lp(ids, zero=None):
    st["zero"] = zero
    r = torch.log_softmax(m(input_ids=ids).logits[0].double(), -1); st["zero"] = None; return r
res = {}
for name, seq, region in (("Ub x4", UBQ * 4, range(76, 152)), ("Ub x1", UBQ, range(0, 76))):
    rng = random.Random(5); mpos = sorted(rng.sample(list(region), int(0.15 * len(region))))
    ids = tok(seq, return_tensors="pt")["input_ids"]; x = ids.clone(); x[0, [p + 1 for p in mpos]] = tok.mask_token_id
    true = ids[0, [p + 1 for p in mpos]]; ix = torch.tensor([p + 1 for p in mpos])
    lp0 = lp(x); g = st["post"]                                         # post-GELU in the masked forward
    load = (g.abs().sum(0)) * wn                                         # summed write magnitude per unit
    order = torch.argsort((load - load[2087]).abs())
    ctl = int(next(u for u in order.tolist() if u != 2087))
    d_unit = (lp(x, 2087)[ix, true] - lp0[ix, true])
    d_ctl = (lp(x, ctl)[ix, true] - lp0[ix, true])
    res[name] = {"masked": len(mpos), "mean_dlogp_true_unit": round(float(d_unit.mean()), 5), "mean_abs_unit": round(float(d_unit.abs().mean()), 5),
                 "control_unit": ctl, "write_load_unit_vs_ctl": [round(float(load[2087]), 2), round(float(load[ctl]), 2)],
                 "mean_dlogp_true_ctl": round(float(d_ctl.mean()), 5), "mean_abs_ctl": round(float(d_ctl.abs().mean()), 5),
                 "unit_value_at_masked_mean": round(float(m.esm.encoder.layer[15].intermediate.dense(torch.zeros(1)).sum()*0), 1) if False else None}
print(json.dumps(res, indent=1)); json.dump(res, open(sys.argv[1], "w"), indent=1)
