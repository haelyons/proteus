#!/bin/bash
# Phase 1 on one GPU. All output under $OUT (persistent filesystem). Env: JOB OUT FS DEADLINE.
set -uo pipefail
echo "[atlas] start $(date -u +%FT%TZ) job=$JOB deadline=$DEADLINE ($(( DEADLINE - $(date +%s) ))s left)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
export HF_HOME=$FS/cache/hf PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Lambda Stack 22.04 ships Pillow without PIL.Image.Resampling, which transformers>=5 imports eagerly
python3 -m pip install --user -q -U "pillow>=10" "transformers>=4.46" 2>&1 | tail -1
python3 -c "import torch, triton, transformers, PIL; from transformers import EsmModel, EsmTokenizer; print('[atlas] versions torch', torch.__version__, 'triton', triton.__version__, 'transformers', transformers.__version__, 'pillow', PIL.__version__, '| EsmModel import ok')" || { echo "[atlas] preflight FAILED: cannot import EsmModel"; exit 2; }
DATA=$FS/cache/swissprot_2026_03.pt
if [ ! -f $DATA ]; then
  echo "[atlas] building Swiss-Prot cache (upload missing)"; python3 scripts/build_swissprot.py --out $DATA > $OUT/swissprot_build.json
fi
echo "[atlas] gpu tests"; python3 -m tests.test_triton_kernel --cuda > $OUT/gpu_tests.log 2>&1; echo "[atlas] gpu tests rc=$? $(grep -E 'kernel|OK|FAIL' $OUT/gpu_tests.log | tr '\n' ' ' | cut -c1-600)"
echo "[atlas] bench"; python3 scripts/bench_phase1.py --model 650M --data $DATA --out $OUT --n-batches 40 --deadline $DEADLINE; rc=$?
echo "[atlas] end $(date -u +%FT%TZ) bench rc=$rc"; exit $rc
