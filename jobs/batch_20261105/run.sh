#!/bin/bash
# One instance, four parts (predictions: results/batch_20261105_predictions.md):
#   A  650M tie audit: filtered top-K vs baukit's algorithm on the first 60 batches, canonical comparison
#      (measurement #1 left ids "differ at 56% of slots" unexplained at 650M; the 8M bf16 check put every
#      difference in the tie group at the K-th value)
#   B  #8 K sweep at 650M: K = 16 / 64 / 256, complete workload, first 200 batches with full-stream seeds
#      (--stop-after, as the gate runs; not --max-batches, which seeds for a 200-batch stream)
#   C  8M atlases for #4 / #6 / #7: pretrained (alone, timed), its held-out atlas on the same edges, and three
#      shuffled-weight controls (--control-seed 1/2/3; same stream, hashes and plan) run concurrently
#   D  #7 ubiquitin share of each unit's samples: 650M atlas #2 (if on this filesystem) and the 8M atlases
# Env: JOB OUT FS DEADLINE; SMOKE=1: 8M on CPU, a few batches per part.
set -uo pipefail
SMOKE=${SMOKE:-0}
PARTS=${PARTS:-$(cat parts.txt 2>/dev/null || echo ABCD)}   # which parts run (parts.txt in the job dir)
on() { [[ $PARTS == *$1* ]]; }
HELD=heldout_created_after_2021-11-17.tsv
EDGES="0.5 0.9 0.99 0.999 0.9999 0.99999"
if [ "$SMOKE" = 1 ]; then
  M650=8M; DEV=cpu; BACKEND=torch; TPB=4096; PT=8192; PTK=4096; NB=4; NK=3; KS="16 64"; NB8=4; PAR=1
  export OMP_NUM_THREADS=1
  DATA=data/swissprot_2026_03.pt; HELD=lists/$HELD; ATLAS650=${SMOKE_ATLAS650:-}
  LIM8="--max-batches $NB8"
else
  M650=650M; DEV=cuda; BACKEND=triton; TPB=32768; PT=262144; PTK=32768; NB=60; NK=200; KS="16 64 256"; PAR=3
  DATA=$FS/cache/swissprot_2026_03.pt; ATLAS650=$FS/runs/atlas-fu650-20261104-1425/full/atlas.pt
  LIM8=""
fi
echo "[atlas] start $(date -u +%FT%TZ) job=$JOB smoke=$SMOKE parts=$PARTS ($(( DEADLINE - $(date +%s) ))s left)"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ "$SMOKE" != 1 ]; then
  export HF_HOME=$FS/cache/hf
  nvidia-smi --query-gpu=name,memory.total,power.limit,clocks.max.sm --format=csv,noheader
  python3 -m pip install --user -q -U "pillow>=10" "transformers>=4.46" 2>&1 | tail -1
  python3 -c "import torch, triton, transformers; from transformers import EsmModel; print('[atlas] versions torch', torch.__version__, 'triton', triton.__version__, 'transformers', transformers.__version__)" || { echo "[atlas] preflight FAILED"; exit 2; }
fi
summ() { python3 - "$@" <<'PY'
import json, statistics as st, sys
for p in sys.argv[1:]:
    try:
        R = [json.loads(l) for l in open(p + "/batches.jsonl")]
    except FileNotFoundError:
        print(f"[atlas]   {p.split('/')[-1]}: no batches"); continue
    o = [r["overhead_vs_fwd"] for r in R if r.get("overhead_vs_fwd") is not None and not r.get("profiled")]
    late = o[len(o) // 2:]
    rd = sum(1 for r in R if r.get("redo_layers"))
    if o: print(f"[atlas]   {p.split('/')[-1]}: {len(R)} batches, overhead mean {st.mean(o):.3f} median {st.median(o):.3f}; second half mean {st.mean(late):.3f} median {st.median(late):.3f}; redo batches {rd}")
    else: print(f"[atlas]   {p.split('/')[-1]}: {len(R)} batches (no paired forwards)")
PY
}
C650="--model $M650 --data $DATA --device $DEV --dtype bf16 --backend $BACKEND --tokens-per-batch $TPB --edge-q $EDGES --exclude-acc $HELD --deadline $DEADLINE --pair-every 1 --ckpt-every 100000"

if on A; then
echo "[atlas] A: tie audit, $M650, first $NB batches"
python3 scripts/run_atlas.py $C650 --K 64 --Kr 64 --pilot-tokens $PTK --max-batches $NB --seed-thresholds --no-reservoir --no-subsample --out $OUT/tie_ours > $OUT/tie_ours.log 2>&1 || echo "[atlas] tie_ours rc=$?"
python3 scripts/topk_baseline.py --model $M650 --data $DATA --device $DEV --dtype bf16 --tokens-per-batch $TPB --exclude-acc $HELD --max-batches $NB --K 64 \
  --compare $OUT/tie_ours/atlas.pt --out $OUT/tie_baukit 2>&1 | tail -1 | sed 's/^/[atlas] tie audit: /'
rm -f $OUT/tie_ours/atlas.pt
fi

if on B; then
echo "[atlas] B: K sweep ($KS), complete workload, first $NK batches, full-stream seeds"
for K in $KS; do
  python3 scripts/run_atlas.py $C650 --K $K --Kr 64 --pilot-tokens $PT --seed-thresholds --stop-after $NK --out $OUT/k$K > $OUT/k$K.log 2>&1 || echo "[atlas] k$K rc=$?"
done
summ $(for K in $KS; do echo $OUT/k$K; done)
fi

if on C; then
echo "[atlas] C: 8M atlases (K 64, Kr 256)"
C8="--model 8M --data $DATA --device $DEV --dtype bf16 --backend $BACKEND --tokens-per-batch $TPB --K 64 --Kr 256 --edge-q $EDGES --deadline $DEADLINE --ckpt-every 2000 $LIM8"
python3 scripts/run_atlas.py $C8 --exclude-acc $HELD --pilot-tokens $PT --seed-thresholds --pair-every 50 --out $OUT/a8_pre > $OUT/a8_pre.log 2>&1 || echo "[atlas] a8_pre rc=$?"
grep -h "done:\|seeded" $OUT/a8_pre.log | sed 's/^/[atlas]   /' | cut -c1-300
python3 scripts/run_atlas.py --model 8M --data $DATA --device $DEV --dtype bf16 --backend $BACKEND --tokens-per-batch $TPB --K 64 --Kr 256 --deadline $DEADLINE \
  --ckpt-every 100000 --only-acc $HELD --edges-from $OUT/a8_pre/atlas.pt $LIM8 --out $OUT/a8_held > $OUT/a8_held.log 2>&1 || echo "[atlas] a8_held rc=$?"
pids=()
for s in 1 2 3; do
  python3 scripts/run_atlas.py $C8 --exclude-acc $HELD --control shuffled --control-seed $s --pilot-tokens $PT --seed-thresholds --out $OUT/a8_shuf$s > $OUT/a8_shuf$s.log 2>&1 &
  pids+=($!)
  [ ${#pids[@]} -ge $PAR ] && { wait ${pids[0]}; pids=("${pids[@]:1}"); }
done
wait
for d in a8_pre a8_held a8_shuf1 a8_shuf2 a8_shuf3; do
  python3 -c "import json; r=json.load(open('$OUT/$d/run.json')); s=r.get('seconds', {}); print('[atlas]   $d: complete', r.get('complete'), 'n_seen', r.get('n_seen'), 'stream_s', round(s.get('stream_this_process', 0), 1), 'GB', round(r.get('atlas_pt_GB', 0), 3))" 2>&1 | cut -c1-300
done
summ $OUT/a8_pre
# slim copies for download (~0.21 GB each): token ids int32, reservoir values fp16, no reservoir keys, Gram fp32
for d in a8_pre a8_held a8_shuf1 a8_shuf2 a8_shuf3; do
  python3 - $OUT/$d/atlas.pt $OUT/$d/slim.pt <<'EOF' 2>&1 | sed "s/^/[atlas]   slim $d: /"
import os, sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cast = {"top_tok": torch.int32, "res_tok": torch.int32, "res_val": torch.float16, "enr": torch.int32, "gram": torch.float32}
assert int(a["res_tok"].max()) < 2 ** 31 and int(a["top_tok"].max()) < 2 ** 31
s = {k: (v.to(cast[k]) if k in cast else v) for k, v in a.items() if k != "res_key"}
torch.save(s, sys.argv[2])
print(round(os.path.getsize(sys.argv[2]) / 1e9, 3), "GB")
EOF
done
fi

if on D; then
echo "[atlas] D: ubiquitin share of samples"
python3 scripts/ubiquitin_audit.py --data $DATA --atlas $OUT/a8_pre/atlas.pt --exclude-acc $HELD --out $OUT/ubq_8m 2>&1 | tail -1 | cut -c1-400 | sed 's/^/[atlas]   8M: /'
for s in 1 2 3; do
  python3 scripts/ubiquitin_audit.py --data $DATA --atlas $OUT/a8_shuf$s/atlas.pt --exclude-acc $HELD --out $OUT/ubq_8m_shuf$s 2>&1 | tail -1 | cut -c1-200 | sed "s/^/[atlas]   8M shuffled $s: /"
done
if [ -n "$ATLAS650" ] && [ -f "$ATLAS650" ]; then
  python3 scripts/ubiquitin_audit.py --data $DATA --atlas $ATLAS650 --exclude-acc $HELD --out $OUT/ubq_650m 2>&1 | tail -1 | cut -c1-400 | sed 's/^/[atlas]   650M: /'
else
  echo "[atlas]   650M atlas not on this filesystem: skipped"
fi
fi
grep -hE "Traceback|Error" $OUT/*.log 2>/dev/null | head -5 | sed 's/^/[atlas] log: /'
echo "[atlas] end $(date -u +%FT%TZ)"
