#!/bin/bash
# 650M L15/2087 use study, fresh run (plan and frozen predictions: results/ubq650_use_plan.md).
#   1. GPU smoke: the one-base discovery smoke constructs, compared with the same forwards on CPU (shipped);
#  1b. the discovery intervention rerun with the amended controls (recorded as discovery);
#   2. fresh: behaviour, screen, intervention on constructs_fresh.json with the frozen layer-15 group;
#   3. analysis (also recomputed locally from the raw files).
# Env: JOB OUT FS DEADLINE. SMOKE=1: run locally on CPU with the smoke constructs only.
set -uo pipefail
SMOKE=${SMOKE:-0}
export PYTHONUNBUFFERED=1
if [ "$SMOKE" = 1 ]; then
  DEV=cpu; FRESH=cons_smoke.json; DISC=cons_smoke.json
else
  export HF_HOME=$FS/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  # Lambda Stack 22.04 ships Pillow without PIL.Image.Resampling, which transformers>=5 imports eagerly
  python3 -m pip install --user -q -U "pillow>=10" "transformers>=4.46" 2>&1 | tail -1
  python3 -c "import torch, transformers; from transformers import EsmForMaskedLM; print('[atlas] versions torch', torch.__version__, 'transformers', transformers.__version__, 'gpu', torch.cuda.get_device_name(0))" || { echo "[atlas] preflight FAILED"; exit 2; }
  DEV=cuda; FRESH=constructs_fresh.json; DISC=constructs_discovery.json
fi
echo "[atlas] start $(date -u +%FT%TZ) job=$JOB smoke=$SMOKE ($(( DEADLINE - $(date +%s) ))s left)"
echo "[atlas] 1: smoke on $DEV"
python3 scripts/ubq650_use.py run --constructs cons_smoke.json --units ubq650_units.json --device $DEV --out $OUT/smoke 2>&1 | grep -v "Loading weights" | sed 's/^/[atlas]   /'
python3 - $OUT/smoke cpu_smoke <<'PY' 2>&1 | sed 's/^/[atlas]   /'
import json, sys, torch
a = torch.load(f"{sys.argv[1]}/behaviour.pt", weights_only=False); b = torch.load(f"{sys.argv[2]}/behaviour.pt", weights_only=False)
dlp = max(float((x["lp"] - y["lp"]).abs().max()) for x, y in zip(a["recs"], b["recs"]))
du = max(float((x["u2087"] - y["u2087"]).abs().max()) for x, y in zip(a["recs"], b["recs"]))
ai = torch.load(f"{sys.argv[1]}/intervention.pt", weights_only=False); bi = torch.load(f"{sys.argv[2]}/intervention.pt", weights_only=False)
dv = max(float((x["lp_var"] - y["lp_var"]).abs().max()) for x, y in zip(ai["recs"], bi["recs"]))
same = all(x["unit_controls"] == y["unit_controls"] for x, y in zip(ai["recs"], bi["recs"]))
r = {"max_abs_dlogp_behaviour": dlp, "max_abs_d_u2087": du, "max_abs_dlogp_intervention": dv, "same_unit_controls": same,
     "same_group": ai["group"] == bi["group"]}
json.dump(r, open(f"{sys.argv[1]}/gpu_vs_cpu.json", "w")); print("gpu vs cpu", r)
PY
echo "[atlas] 1b: discovery intervention again, with the controls amended after the pilot"
python3 scripts/ubq650_use.py run --constructs $DISC --units ubq650_units.json --parts intervention --group group_frozen.json --device $DEV --out $OUT/disc_iv 2>&1 \
  | grep -v "Loading weights" | sed 's/^/[atlas]   /'
echo "[atlas] 2: fresh run"
timeout $(( DEADLINE - $(date +%s) - 600 )) python3 scripts/ubq650_use.py run --constructs $FRESH --units ubq650_units.json --group group_frozen.json --device $DEV --out $OUT/fresh 2>&1 \
  | grep -v "Loading weights" | sed 's/^/[atlas]   /'
echo "[atlas] 3: analysis"
python3 scripts/ubq650_use_analyse.py --constructs $FRESH --run $OUT/fresh --out $OUT/fresh_analysis > $OUT/fresh_analysis.log 2>&1; echo "[atlas]   analysis rc=$?"
echo "[atlas] end $(date -u +%FT%TZ)"
