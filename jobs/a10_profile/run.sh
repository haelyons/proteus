#!/bin/bash
# A10 profiling job. Env: JOB OUT FS DEADLINE. SMOKE=1 runs the same steps with ESM-2 8M on CPU
# (standing policy: nothing launches that has not run end to end somewhere).
#   1. GPU tests (kernel incl. early-exit cases, exactness both backends)
#   2. forward profile: kernel classes + dtypes, determinism, GEMM microbench, Triton tile sweep
#   3. seeded short run (120 batches): warm-up with exact-preserving seeds
#   4. unseeded long run (620 batches, past 16M tokens): overhead decay, checkpoints, far-end profile
# nvidia-smi samples power / SM clock / throttle reasons every 250 ms throughout (smi.csv).
set -uo pipefail
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = 1 ]; then
  MODEL=8M; DEV=cpu; DT=fp32; BACKEND=torch; NB_SEED=4; NB_LONG=8; PROF_AT=6; TPB=4096; PT=8192; NPROF=8; CKPT=3
else
  MODEL=650M; DEV=cuda; DT=bf16; BACKEND=triton; NB_SEED=120; NB_LONG=620; PROF_AT=600; TPB=32768; PT=131072; NPROF=12; CKPT=200
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
  F=""
  for cand in "clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown,clocks_event_reasons.sw_thermal_slowdown" \
              "clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.sw_thermal_slowdown"; do
    nvidia-smi --query-gpu=$cand --format=csv,noheader -c 1 >/dev/null 2>&1 && { F=",$cand"; break; }
  done
  Q="timestamp,power.draw,power.limit,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,utilization.gpu$F"
  echo "$Q" > $OUT/smi.csv
  nvidia-smi --query-gpu=$Q --format=csv,noheader,nounits -lms 250 >> $OUT/smi.csv 2>&1 &
  SMI_PID=$!
  echo "[atlas] smi logger pid $SMI_PID fields: $Q"
  echo "[atlas] gpu tests"; python3 -m tests.test_triton_kernel --cuda > $OUT/gpu_tests.log 2>&1
  echo "[atlas] gpu tests rc=$? $(grep -cE 'kernel OK' $OUT/gpu_tests.log) kernel cases OK; $(grep -E '\] (OK|FAILED)' $OUT/gpu_tests.log | tr '\n' ' ' | cut -c1-400)"
else
  DATA=${SMOKE_DATA:-data/swissprot_test500.pt}
fi
if [ ! -f $DATA ]; then echo "[atlas] missing $DATA"; exit 3; fi

echo "[atlas] profile_forward"
python3 scripts/profile_forward.py --model $MODEL --data $DATA --out $OUT --device $DEV --n-batches $NPROF --tokens-per-batch $TPB \
  --sustain-s $([ "$SMOKE" = 1 ] && echo 1 || echo 15) --probe-layer $([ "$SMOKE" = 1 ] && echo 3 || echo 16) || echo "[atlas] profile_forward rc=$?"

COMMON="--model $MODEL --data $DATA --device $DEV --dtype $DT --backend $BACKEND --tokens-per-batch $TPB --pilot-tokens $PT --pair-forward --deadline $DEADLINE"
echo "[atlas] seeded run ($NB_SEED batches)"
python3 scripts/run_atlas.py $COMMON --out $OUT/seeded --max-batches $NB_SEED --seed-thresholds --ckpt-every 100000 || echo "[atlas] seeded rc=$?"
echo "[atlas] long run ($NB_LONG batches, unseeded)"
python3 scripts/run_atlas.py $COMMON --out $OUT/long --max-batches $NB_LONG --ckpt-every $CKPT --profile-at $PROF_AT; rc=$?
[ -n "$SMI_PID" ] && kill $SMI_PID 2>/dev/null
echo "[atlas] end $(date -u +%FT%TZ) long rc=$rc"
exit $rc
