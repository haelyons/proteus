#!/bin/bash
# Runs after followup_650m on the same instance: whole-build descriptor summary (status counts,
# disorder fractions, held-out subset) from the descriptor file on the filesystem.
set -uo pipefail
PREV=$FS/runs/atlas-fu650-20261104-1425
echo "[atlas] post: previous job tail:"; tail -4 $PREV/job.log | sed 's/^/[atlas]   /'
python3 - <<PY 2>&1 | sed 's/^/[atlas] /'
import torch, json, collections
d = torch.load("$FS/descriptors_swissprot_2026_03.pt", weights_only=False)
st = collections.Counter(d["status"].values()); n = sum(st.values())
held = {ln.split("\t")[0] for ln in open("heldout_created_after_2021-11-17.tsv") if ln.strip() and not ln.startswith("Entry")}
p = d["plddt"].float(); ss = d["ss3"]
off = d["offset"]; accs = d["acc"]
hmask = torch.zeros(p.numel(), dtype=torch.bool)
for i, a in enumerate(accs):
    if a in held: hmask[off[i]:off[i + 1]] = True
def frac(m): return {"lt50": round(float((p[m] < 50).float().mean()), 4), "50to70": round(float(((p[m] >= 50) & (p[m] < 70)).float().mean()), 4), "lt70": round(float((p[m] < 70).float().mean()), 4), "ge90": round(float((p[m] >= 90).float().mean()), 4), "residues": int(m.sum())}
out = {"accessions": n, "status": dict(st), "frac_status": {k: round(v / n, 5) for k, v in st.items()},
       "plddt_all": frac(torch.ones_like(hmask)), "plddt_heldout": frac(hmask), "plddt_train": frac(~hmask),
       "ss3_frac_plddt_ge70": [round(float((ss[p >= 70] == k).float().mean()), 4) for k in range(3)],
       "heldout_ok": sum(1 for a, s in d["status"].items() if a in held and s == "ok"), "heldout_listed": len(held)}
json.dump(out, open("$OUT/descriptor_summary.json", "w"), indent=1)
print(json.dumps(out))
PY
cp $PREV/smi.csv $OUT/smi_followup.csv 2>/dev/null; cp $PREV/ps.log $OUT/ps_followup.log 2>/dev/null
echo "[atlas] post end $(date -u +%FT%TZ)"
