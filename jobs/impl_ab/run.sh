#!/bin/bash
# Implementation A/B on one GPU. Env: JOB OUT FS DEADLINE. SMOKE=1 runs the same steps with ESM-2 8M
# on CPU (standing policy: nothing launches that has not run end to end somewhere).
#   1. GPU tests (Triton kernel; exactness on CUDA incl. the int8 / out_dtype GEMM paths)
#   2. GEMM / read-floor microbench at the per-layer subsample shape (bench_gram.py)
#   3. A = reference (stats v1, merge loop): seeded, 620-batch plan, stop at 160 with a checkpoint,
#      profile batch 150 (kernel table + per-range trace attribution)
#   4. B = new (stats v2, merge stacked): same plan and seeds, stop at 160 with a checkpoint, profile 150
#   5. compare A vs B checkpoints at batch 160 (exact fields bitwise; sums by max rel diff)
#   6. resume B to 620 (also a GPU resume test at scale), profile batch 600: steady-state overhead
# nvidia-smi samples power / SM clock every 250 ms (smi.csv).
set -uo pipefail
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = 1 ]; then
  # plan of 400 batches (seeds from n_total ~1.6M tokens, so the filtered path runs as on the GPU), stop early
  MODEL=8M; DEV=cpu; DT=fp32; BACKEND=torch; NB=400; STOP=6; PA=5; PB=10; TPB=4096; PT=65536; RSTOP="--stop-after 6"
  BENCH="--n 200 --D 320 --A 144 --reps 3"
  export OMP_NUM_THREADS=1  # multi-threaded CPU SDPA is not bitwise reproducible on padded batches; A vs B must be
else
  MODEL=650M; DEV=cuda; DT=bf16; BACKEND=triton; NB=620; STOP=160; PA=150; PB=600; TPB=32768; PT=131072; RSTOP=""
  BENCH="--n 1600 --D 5120 --A 144 --reps 20"
fi
echo "[atlas] start $(date -u +%FT%TZ) job=$JOB smoke=$SMOKE deadline=$DEADLINE ($(( DEADLINE - $(date +%s) ))s left)"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA=$FS/cache/swissprot_2026_03.pt
SMI_PID=""
if [ "$SMOKE" != 1 ]; then
  export HF_HOME=$FS/cache/hf
  nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit,clocks.max.sm --format=csv,noheader
  # Lambda Stack 22.04 ships Pillow without PIL.Image.Resampling, which transformers>=5 imports eagerly
  python3 -m pip install --user -q -U "pillow>=10" "transformers>=4.46" 2>&1 | tail -1
  python3 -c "import torch, triton, transformers, PIL; from transformers import EsmModel; print('[atlas] versions torch', torch.__version__, 'triton', triton.__version__, 'transformers', transformers.__version__, 'pillow', PIL.__version__, '| EsmModel import ok')" || { echo "[atlas] preflight FAILED"; exit 2; }
  Q="timestamp,power.draw,power.limit,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,utilization.gpu"
  echo "$Q" > $OUT/smi.csv
  nvidia-smi --query-gpu=$Q --format=csv,noheader,nounits -lms 250 >> $OUT/smi.csv 2>&1 &
  SMI_PID=$!
  echo "[atlas] gpu tests"; python3 -m tests.test_triton_kernel --cuda > $OUT/gpu_tests.log 2>&1
  echo "[atlas] gpu tests rc=$? $(grep -cE 'kernel OK' $OUT/gpu_tests.log) kernel cases OK; $(grep -E '\] (OK|FAILED)' $OUT/gpu_tests.log | tr '\n' ' ' | cut -c1-500)"
else
  DATA=${SMOKE_DATA:-data/swissprot_2026_03.pt}
fi
if [ ! -f $DATA ]; then echo "[atlas] missing $DATA"; exit 3; fi

echo "[atlas] bench_gram"
python3 scripts/bench_gram.py $BENCH --out $OUT/bench_gram.json 2>&1 | sed 's/^/[atlas]   /' || echo "[atlas] bench_gram rc=$?"

COMMON="--model $MODEL --data $DATA --device $DEV --dtype $DT --backend $BACKEND --tokens-per-batch $TPB --pilot-tokens $PT
        --max-batches $NB --seed-thresholds --pair-forward --deadline $DEADLINE --ckpt-every $STOP"
echo "[atlas] A: stats v1, merge loop (stop at $STOP)"
python3 scripts/run_atlas.py $COMMON --out $OUT/A --stats-impl v1 --merge-impl loop --stop-after $STOP --profile-at $PA \
  || echo "[atlas] A rc=$?"
echo "[atlas] B: stats v2, merge stacked (stop at $STOP)"
python3 scripts/run_atlas.py $COMMON --out $OUT/B --stats-impl v2 --merge-impl stacked --stop-after $STOP --profile-at $PA \
  || echo "[atlas] B rc=$?"
echo "[atlas] compare A vs B at batch $STOP"
python3 scripts/compare_atlas.py $OUT/A/ckpt/ckpt_$STOP.pt $OUT/B/ckpt/ckpt_$STOP.pt --out $OUT/compare_A_B.json > $OUT/compare.log 2>&1
echo "[atlas] compare rc=$? $(python3 -c "import json; r=json.load(open('$OUT/compare_A_B.json')); print('ok', r['ok'], {k: v for k, v in r['fields'].items() if not (v.get('exact') or v.get('bitwise') or v.get('equal'))})" 2>&1 | cut -c1-600)"
for X in A B; do
  echo "[atlas] $X batch $PA ranges:"; sed -n 1,40p $OUT/$X/profile_batch_${PA}_ranges.txt 2>/dev/null | grep -E "^(mode|range|atlas)" | sed 's/^/[atlas]   /'
done
echo "[atlas] B resumed to $NB (profile $PB) $RSTOP"
python3 scripts/run_atlas.py $COMMON --out $OUT/B --stats-impl v2 --merge-impl stacked --resume --profile-at $PB $RSTOP; rc=$?
echo "[atlas] B batch $PB ranges:"; grep -E "^(mode|range|atlas)" $OUT/B/profile_batch_${PB}_ranges.txt 2>/dev/null | sed 's/^/[atlas]   /'
python3 - <<EOF 2>&1 | sed 's/^/[atlas] /'
import json, statistics as st
for X in ("A", "B"):
    R = [json.loads(l) for l in open("$OUT/" + X + "/batches.jsonl")]
    seen = {}
    for r in R: seen[r["batch"]] = r          # replayed batches after resume: keep the last
    for lo, hi in ((0, 120), (100, $STOP), (500, $NB)):
        rr = [r for b, r in sorted(seen.items()) if lo <= b < hi and r.get("overhead_vs_fwd") is not None]
        if not rr: continue
        o = [r["overhead_vs_fwd"] for r in rr]
        pooled = sum(r["atlas_pass_s"] for r in rr) / sum(r["fwd_only_s"] for r in rr) - 1
        q = [r["overhead_vs_fwd"] for r in rr if not (r["redo_layers"] or r["profiled"] or r.get("gram_flushed"))]
        print(f"{X} batches {lo}-{hi}: n={len(rr)} overhead mean {st.mean(o):.3f} median {st.median(o):.3f} pooled {pooled:.3f}; "
              f"excl. redo/profiled/gram-flush: n={len(q)} mean {st.mean(q) if q else float('nan'):.3f}; "
              f"fwd {st.mean(r['fwd_only_s'] for r in rr):.3f}s; redo batches {sum(1 for r in rr if r['redo_layers'])}; "
              f"bulk rows kept {sum(r.get('bulk_rows_kept', 0) for r in rr)}/{sum(r.get('bulk_rows_in', 0) for r in rr)}")
EOF
[ -n "$SMI_PID" ] && kill $SMI_PID 2>/dev/null
echo "[atlas] end $(date -u +%FT%TZ) rc=$rc"
exit $rc
