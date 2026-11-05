import sys, json, random, torch
sys.path.insert(0, "/home/user/proteus")
from atlas.esm import load
UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
torch.set_grad_enabled(False)
model, tok, _ = load("650M", "none", dtype=torch.float32, device="cpu")
cap = {}
model.encoder.layer[15].intermediate.dense.register_forward_hook(lambda _m, _i, o: cap.__setitem__("h", o[0, :, 2087]))
def acts(s):
    model(**tok(s[:1022], return_tensors="pt")); return cap["h"][1:-1].clone()
res = {"ubiquitin_copies": {}}
for k in (1, 2, 3, 4, 6, 9):
    v = acts(UBQ * k); h = [float(v[76 * c + 67]) for c in range(k)]
    res["ubiquitin_copies"][k] = {"copy1": round(h[0], 3), "median": round(sorted(h)[len(h) // 2], 3)}
d = torch.load("/home/user/proteus/data/swissprot_2026_03.pt", weights_only=False)
held = [l.split("\t")[0] for l in open("/home/user/proteus/lists/heldout_created_after_2021-11-17.tsv") if l.strip() and not l.startswith("Entry")]
idx = {a: i for i, a in enumerate(d["acc"])}
rng = random.Random(3); frags = []
for a in rng.sample([x for x in held if x in idx and len(d["seq"][idx[x]]) >= 200], 5):
    s = d["seq"][idx[a]]; p = rng.randint(0, len(s) - 80); f = s[p:p + 80]
    frags.append({"acc": a, "max": {k: round(float(acts(f * k).max()), 3) for k in (1, 2, 4)}})
res["heldout_fragments"] = frags
res["monotone_1_2_4"] = sum(1 for f in frags if f["max"][1] < f["max"][2] < f["max"][4])
print(json.dumps(res, indent=1)); json.dump(res, open(sys.argv[1], "w"), indent=1)
