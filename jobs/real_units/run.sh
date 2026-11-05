#!/bin/bash
# Round 2 inputs (plan: results/cases_round2_plan.md):
#   1. frozen ruler v1.2 over every positive-pole column of the 8M atlas and its three shuffled controls, with
#      descriptors for the whole stream (scripts/score_real_units.py; BH across each atlas's columns);
#   2. the 650M ubiquitin-enriched units (>= half of top-64 in ubiquitin copies, plus the 40 largest shares):
#      their top-64 and reservoir samples from atlas #2, exported as JSON.
# Env: JOB OUT FS DEADLINE; SMOKE=1: local descriptor parts, local slim atlases, a few columns.
set -uo pipefail
SMOKE=${SMOKE:-0}
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [ "$SMOKE" = 1 ]; then
  DATA=data/swissprot_2026_03.pt; DESC=$SMOKE_DESC; UNIREF=lists/swissprot_2026_03_uniref50.tsv.gz
  A_PRE=$SMOKE_A8/a8_pre.slim.pt; A_S1=$SMOKE_A8/a8_shuf1.slim.pt; A_S2=$SMOKE_A8/a8_shuf2.slim.pt; A_S3=$SMOKE_A8/a8_shuf3.slim.pt
  UBQ=$SMOKE_UBQ; A650=$SMOKE_A8/a8_pre.slim.pt; W=3; MAXC="--max-columns 64"; B=300
else
  DATA=$FS/cache/swissprot_2026_03.pt; DESC=$FS/descriptors_swissprot_2026_03.pt; UNIREF=swissprot_2026_03_uniref50.tsv.gz
  A_PRE=$FS/runs/atlas-batch-20261105-0258/a8_pre/slim.pt; A_S2=$FS/runs/atlas-batch-20261105-0258/a8_shuf2/slim.pt
  A_S1=$FS/runs/atlas-batch2-20261105-0348/a8_shuf1/slim.pt; A_S3=$FS/runs/atlas-batch2-20261105-0348/a8_shuf3/slim.pt
  UBQ=$FS/runs/atlas-batch-20261105-0258/ubq_650m/ubq_fractions.pt; A650=$FS/runs/atlas-fu650-20261104-1425/full/atlas.pt
  W=$(( $(getconf _NPROCESSORS_ONLN) - 2 )); MAXC=""; B=2000
fi
echo "[atlas] start $(date -u +%FT%TZ) job=$JOB smoke=$SMOKE ($(( DEADLINE - $(date +%s) ))s left); $(getconf _NPROCESSORS_ONLN) cpus; $(free -g | awk '/Mem/{print $2}') GB RAM"
echo "[atlas] 1: ruler v1.2 over the 8M atlases"
timeout $(( DEADLINE - $(date +%s) - 900 )) python3 scripts/score_real_units.py --desc $DESC --data $DATA --uniref $UNIREF \
  --atlas $A_PRE $A_S1 $A_S2 $A_S3 --names pre shuf1 shuf2 shuf3 --out $OUT --workers $W --B $B $MAXC 2>&1 | grep -v Warning | sed 's/^/[atlas]   /' | cut -c1-400
echo "[atlas] 2: 650M ubiquitin-enriched units"
python3 - $UBQ $A650 $OUT/ubq650_units.json <<'PY' 2>&1 | sed 's/^/[atlas]   /'
import json, sys, torch
fr = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["frac_top"].float()   # [L, 2, D]
L, P, D = fr.shape
flat = fr.flatten()
order = torch.argsort(flat, descending=True)
keep = sorted(set(torch.nonzero(flat >= 0.5).flatten().tolist()) | set(order[:40].tolist()))
at = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
out = []
for k in keep:
    l, p, u = k // (P * D), (k // D) % P, k % D
    out.append({"layer": l, "pole": p, "unit": u, "ubq_top_share": round(float(fr[l, p, u]), 4),
                "top_tok": at["top_tok"][l, p, u].tolist(), "top_val": [round(float(v), 4) for v in at["top_val"][l, p, u]],
                "edges": [round(float(v), 4) for v in at["edges"][l, p, u]],
                "res_tok": at["res_tok"][l, p, u].tolist()})
json.dump(out, open(sys.argv[3], "w"))
print(len(out), "units exported;", sum(1 for o in out if o["ubq_top_share"] >= 0.5), "with >= half; best", max(o["ubq_top_share"] for o in out))
PY
echo "[atlas] end $(date -u +%FT%TZ)"
