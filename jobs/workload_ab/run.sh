#!/bin/bash
# Measurement #1: the complete collection workload done three ways on the same batches, outputs checked.
#   filtered = this project (seeded thresholds, compaction, merges); naive = exact dense selection for every
#   layer and batch (run_atlas --naive); baukit = baukit's TopK algorithm (re-implemented, top-K only).
# Prefix (first NB batches of the frozen plan, all paths fresh): complete filtered vs complete naive
# (atlases compared field by field); top-K only: filtered vs naive vs baukit (values compared).
# Steady state (batches 4500.., if the follow-up checkpoint is on this filesystem): filtered resumed from
# its batch-4500 checkpoint vs naive and baukit on the same batches with fresh state (timing only).
# Also exports a descriptor subset (67k proteins) for the ruler work. Env: JOB OUT FS DEADLINE; SMOKE=1: 8M, CPU.
set -uo pipefail
SMOKE=${SMOKE:-0}
HELD=heldout_created_after_2021-11-17.tsv
EDGES="0.5 0.9 0.99 0.999 0.9999 0.99999"
if [ "$SMOKE" = 1 ]; then
  MODEL=8M; DEV=cpu; DT=fp32; BACKEND=torch; TPB=4096; PT=8192; PTK=4096; NB=4; NS=3; OFF=50
  export OMP_NUM_THREADS=1
  DATA=data/swissprot_2026_03.pt; HELD=lists/$HELD; DESC=${SMOKE_DESC:-}; FUCK=${SMOKE_CKPT:-}
  NSUB=1500
else
  MODEL=650M; DEV=cuda; DT=bf16; BACKEND=triton; TPB=32768; PT=262144; PTK=32768; NB=60; NS=60; OFF=4500
  DATA=$FS/cache/swissprot_2026_03.pt; DESC=$FS/descriptors_swissprot_2026_03.pt
  FUCK=$FS/runs/atlas-fu650-20261104-1425/full/ckpt/ckpt_4500.pt
  NSUB=60000
fi
echo "[atlas] start $(date -u +%FT%TZ) job=$JOB smoke=$SMOKE ($(( DEADLINE - $(date +%s) ))s left)"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ "$SMOKE" != 1 ]; then
  export HF_HOME=$FS/cache/hf
  nvidia-smi --query-gpu=name,memory.total,power.limit,clocks.max.sm --format=csv,noheader
  python3 -m pip install --user -q -U "pillow>=10" "transformers>=4.46" 2>&1 | tail -1
  python3 -c "import torch, triton, transformers; from transformers import EsmModel; print('[atlas] versions torch', torch.__version__, 'triton', triton.__version__, 'transformers', transformers.__version__)" || { echo "[atlas] preflight FAILED"; exit 2; }
fi
if [ -n "$DESC" ] && [ -f "$DESC" ]; then
  echo "[atlas] descriptor subset export"
  python3 - <<PY 2>&1 | sed 's/^/[atlas] /'
import torch, os
d = torch.load("$DESC", weights_only=False)
held = {ln.split("\t")[0] for ln in open("$HELD") if ln.strip() and not ln.startswith("Entry")}
accs = d["acc"]; off = d["offset"]
g = torch.Generator().manual_seed(11)
train = [i for i, a in enumerate(accs) if a not in held]
pick = sorted(torch.tensor(train)[torch.randperm(len(train), generator=g)[:$NSUB]].tolist() + [i for i, a in enumerate(accs) if a in held])
os.makedirs("$OUT/desc_subset", exist_ok=True)
for c in range(0, len(pick), 5000):
    ii = pick[c:c + 5000]
    lens = [int(off[i + 1] - off[i]) for i in ii]
    part = {"acc": [accs[i] for i in ii], "seq_idx": d["seq_idx"][ii], "offset": torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist())),
            "heldout": [accs[i] in held for i in ii]}
    for k in d["fields"]:
        part[k] = torch.cat([d[k][off[i]:off[i + 1]] for i in ii])
    torch.save(part, f"$OUT/desc_subset/part_{c // 5000:02d}.pt")
print(f"exported {len(pick)} proteins ({sum(1 for i in pick if accs[i] in held)} held out) in {(len(pick) + 4999) // 5000} parts")
PY
fi

T="--model $MODEL --data $DATA --device $DEV --dtype $DT --backend $BACKEND --tokens-per-batch $TPB --K 64 --Kr 64 --edge-q $EDGES --exclude-acc $HELD --deadline $DEADLINE --pair-every 1 --ckpt-every 100000"
summ() { python3 - "$@" <<'PY'
import json, statistics as st, sys
for p in sys.argv[1:]:
    try:
        R = [json.loads(l) for l in open(p + "/batches.jsonl")]
    except FileNotFoundError:
        print(f"[atlas]   {p}: no batches"); continue
    o = [r["overhead_vs_fwd"] for r in R if r.get("overhead_vs_fwd") is not None and not r.get("profiled")]
    f = [r["fwd_only_s"] for r in R if r.get("fwd_only_s") is not None]
    if o: print(f"[atlas]   {p.split('/')[-1]}: {len(o)} batches, overhead mean {st.mean(o):.3f} median {st.median(o):.3f}; forward {st.mean(f):.3f}s")
PY
}
echo "[atlas] prefix: first $NB batches, fresh state"
python3 scripts/run_atlas.py $T --pilot-tokens $PT --max-batches $NB --seed-thresholds --out $OUT/pre_filtered > $OUT/pre_filtered.log 2>&1 || echo "[atlas] pre_filtered rc=$?"
python3 scripts/run_atlas.py $T --pilot-tokens $PT --max-batches $NB --naive --out $OUT/pre_naive > $OUT/pre_naive.log 2>&1 || echo "[atlas] pre_naive rc=$?"
python3 scripts/run_atlas.py $T --pilot-tokens $PTK --max-batches $NB --seed-thresholds --no-reservoir --no-subsample --out $OUT/pre_filtered_topk > $OUT/pre_filtered_topk.log 2>&1 || echo "[atlas] pre_filtered_topk rc=$?"
python3 scripts/run_atlas.py $T --pilot-tokens $PTK --max-batches $NB --naive --no-reservoir --no-subsample --out $OUT/pre_naive_topk > $OUT/pre_naive_topk.log 2>&1 || echo "[atlas] pre_naive_topk rc=$?"
python3 scripts/topk_baseline.py --model $MODEL --data $DATA --device $DEV --dtype $DT --tokens-per-batch $TPB --exclude-acc $HELD --max-batches $NB --K 64 \
  --compare $OUT/pre_filtered_topk/atlas.pt --out $OUT/pre_baukit_topk 2>&1 | tail -1 | sed 's/^/[atlas] baukit vs filtered top-K: /'
python3 scripts/compare_atlas.py $OUT/pre_filtered/atlas.pt $OUT/pre_naive/atlas.pt --out $OUT/cmp_complete.json > /dev/null 2>&1; echo "[atlas] complete filtered vs naive: rc=$? $(python3 -c "import json; r=json.load(open('$OUT/cmp_complete.json')); print('ok', r['ok'], {k: v for k, v in r['fields'].items() if not (v.get('exact') or v.get('equal'))})" 2>&1 | cut -c1-400)"
python3 scripts/compare_atlas.py $OUT/pre_filtered_topk/atlas.pt $OUT/pre_naive_topk/atlas.pt --out $OUT/cmp_topk.json > /dev/null 2>&1; echo "[atlas] top-K filtered vs naive: rc=$? $(python3 -c "import json; r=json.load(open('$OUT/cmp_topk.json')); print('ok', r['ok'], {k: v for k, v in r['fields'].items() if k.startswith('top') and not v.get('exact')})" 2>&1 | cut -c1-300)"
echo "[atlas] prefix timings:"; summ $OUT/pre_filtered $OUT/pre_naive $OUT/pre_filtered_topk $OUT/pre_naive_topk $OUT/pre_baukit_topk

if [ -n "$FUCK" ] && [ -f "$FUCK" ]; then
  echo "[atlas] steady state: batches $OFF.. ($NS)"
  mkdir -p $OUT/st_filtered/ckpt && cp $FUCK $OUT/st_filtered/ckpt/
  python3 scripts/run_atlas.py $T --pilot-tokens $PT --seed-thresholds --resume --stop-after $NS --out $OUT/st_filtered > $OUT/st_filtered.log 2>&1 || echo "[atlas] st_filtered rc=$?"
  rm -f $OUT/st_filtered/ckpt/*.pt
  python3 scripts/run_atlas.py $T --pilot-tokens $PT --naive --plan-offset $OFF --stop-after $NS --out $OUT/st_naive > $OUT/st_naive.log 2>&1 || echo "[atlas] st_naive rc=$?"
  python3 scripts/run_atlas.py $T --pilot-tokens $PTK --naive --no-reservoir --no-subsample --plan-offset $OFF --stop-after $NS --out $OUT/st_naive_topk > $OUT/st_naive_topk.log 2>&1 || echo "[atlas] st_naive_topk rc=$?"
  python3 scripts/topk_baseline.py --model $MODEL --data $DATA --device $DEV --dtype $DT --tokens-per-batch $TPB --exclude-acc $HELD --plan-offset $OFF --stop-after $NS --K 64 \
    --out $OUT/st_baukit_topk 2>&1 | tail -1 | sed 's/^/[atlas] baukit steady: /'
  echo "[atlas] steady timings:"; summ $OUT/st_filtered $OUT/st_naive $OUT/st_naive_topk $OUT/st_baukit_topk
else
  echo "[atlas] steady state skipped: no follow-up checkpoint on this filesystem"
fi
grep -hE "Traceback|Error" $OUT/*.log 2>/dev/null | head -5 | sed 's/^/[atlas] log: /'
echo "[atlas] end $(date -u +%FT%TZ)"
