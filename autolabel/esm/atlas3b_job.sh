#!/bin/bash
# On-box job: ESM-2 3B atlas over Swiss-Prot 2026_03 minus the held-out list, with proteus's run_atlas.py and the
# 650M science-pass config (K 64, edges at pilot q 0.5..0.99999, seeded thresholds, stats v3, merge stacked), except:
# pilot 131,072 tokens (host RAM: tokens x 36 x 10240 x 2 B = 97 GB), Kr 16 and 16,384 tokens per batch (GPU memory on a
# 40 GB A100: weights 5.7 GB + Gram 15.1 + reservoir 1.4 + enrichment 2.25 + top-K 0.9; 32,768 per batch ran out of memory).
# Then the first-read bundle (export_firstread.py). Writes everything under $OUT (persistent filesystem).
# Ends with $OUT/DONE or $OUT/FAILED. Env: OUT FS DEADLINE (epoch s) [PROTEUS] [PY] [SMOKE=1 SMOKE_DATA SMOKE_DESC]
set -uo pipefail
PROTEUS=${PROTEUS:-$HOME/proteus}; PY=${PY:-python3}; SMOKE=${SMOKE:-0}; EXPORT=${EXPORT:-$HOME/export_firstread.py}
cd "$PROTEUS" || exit 1
EDGES="0.5 0.9 0.99 0.999 0.9999 0.99999"; HELD=lists/heldout_created_after_2021-11-17.tsv
if [ "$SMOKE" = 1 ]; then
  MODEL=facebook/esm2_t6_8M_UR50D; TAG=8M; DEV=cpu; DT=fp32; BACKEND=torch; FAST="--stats-impl v1 --merge-impl stacked"
  TPBS="4096"; PT=8192; KR=32; GATE=3; MAXB="--max-batches 8"; CKPT=4; FLUSH=4
  DATA=$SMOKE_DATA; DESC=$SMOKE_DESC; export OMP_NUM_THREADS=1
else
  MODEL=facebook/esm2_t36_3B_UR50D; TAG=3B; DEV=cuda; DT=bf16; BACKEND=triton; FAST="--stats-impl v3 --merge-impl stacked"
  TPBS="16384 8192"; PT=131072; KR=16; GATE=20; MAXB=""; CKPT=1500; FLUSH=1000
  DATA=$FS/cache/swissprot_2026_03.pt; DESC=$FS/descriptors_swissprot_2026_03.pt; export HF_HOME=$FS/cache/hf
fi
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
say() { echo "[3b $(date -u +%H:%M:%S)] $*"; }
fin() { date -u +%FT%TZ > "$OUT/$1"; say "$1 ${2:-}"; exit "${3:-0}"; }
say "start smoke=$SMOKE model=$MODEL; $(( DEADLINE - $(date +%s) ))s to deadline; $(nproc) cpus, $(free -g | awk '/Mem/{print $2}') GB RAM"
if [ "$SMOKE" != 1 ]; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  $PY -m pip install --user -q -U "pillow>=10" "transformers>=4.46" 2>&1 | tail -1
  $PY -c "import torch, triton; from transformers import EsmModel; print('preflight: torch', torch.__version__, 'triton', triton.__version__, 'EsmModel ok')" || fin FAILED preflight 2
  $PY -m tests.test_triton_kernel --cuda > "$OUT/gpu_tests.log" 2>&1 || fin FAILED "compaction kernel tests (gpu_tests.log)" 5
  $PY -m tests.test_stats_triton --cuda > "$OUT/stats_kernel_tests.log" 2>&1 || fin FAILED "stats/Gram kernel tests" 5
  say "gpu tests ok"
fi
COMMON="--model $MODEL --data $DATA --device $DEV --dtype $DT --backend $BACKEND --K 64 --Kr $KR --edge-q $EDGES --seed-thresholds --exclude-acc $HELD --deadline $DEADLINE"
oom() { grep -q -i -E 'out of memory|OutOfMemoryError' "$1"; }
TPB=""
for T in $TPBS; do
  say "gate at tokens-per-batch $T: fast ($FAST) vs reference (stats v1, merge loop), $GATE batches, pilot 32768"
  rm -rf "$OUT/gate_A" "$OUT/gate_B"
  $PY scripts/run_atlas.py $COMMON --tokens-per-batch $T --pilot-tokens 32768 --out "$OUT/gate_B" $FAST --stop-after $GATE --ckpt-every $GATE > "$OUT/gate_B.log" 2>&1; rb=$?
  if [ $rb != 0 ] && oom "$OUT/gate_B.log"; then say "fast path OOM at $T"; continue; fi
  $PY scripts/run_atlas.py $COMMON --tokens-per-batch $T --pilot-tokens 32768 --out "$OUT/gate_A" --stats-impl v1 --merge-impl loop --stop-after $GATE --ckpt-every $GATE > "$OUT/gate_A.log" 2>&1; ra=$?
  if [ $ra != 0 ] && oom "$OUT/gate_A.log"; then say "reference path OOM at $T"; continue; fi
  [ $ra = 0 ] && [ $rb = 0 ] || fin FAILED "gate runs rc A=$ra B=$rb (gate_*.log)" 6
  TPB=$T; break
done
[ -n "$TPB" ] || fin FAILED "out of memory at every batch size ($TPBS)" 7
$PY scripts/compare_atlas.py "$OUT/gate_A/ckpt/ckpt_$GATE.pt" "$OUT/gate_B/ckpt/ckpt_$GATE.pt" --out "$OUT/gate_compare.json" > "$OUT/gate_compare.log" 2>&1; gate=$?
say "gate rc=$gate at tokens-per-batch $TPB $($PY -c "import json; r=json.load(open('$OUT/gate_compare.json')); print('ok', r.get('ok'))" 2>&1 | tail -1)"
[ $gate = 0 ] || fin FAILED "equivalence gate (gate_compare.json)" 6
grep -h -i -E 'max_memory|peak|GiB' "$OUT/gate_B.log" | tail -2
rm -rf "$OUT/gate_A/ckpt" "$OUT/gate_B/ckpt"
say "full pass, pilot $PT, Kr $KR, tokens-per-batch $TPB"
t0=$(date +%s)
$PY scripts/run_atlas.py $COMMON --tokens-per-batch $TPB --pilot-tokens $PT --out "$OUT/full" $FAST $MAXB --ckpt-every $CKPT --gram-flush-every $FLUSH > "$OUT/full.log" 2>&1; rc=$?
say "full pass rc=$rc wall $(( $(date +%s) - t0 ))s"
[ -f "$OUT/full/atlas.pt" ] || fin FAILED "full pass incomplete (rc $rc; resume from $OUT/full/ckpt with --resume)" 8
ls -la "$OUT/full/atlas.pt"
$PY "$EXPORT" bundle --model $TAG --hf-model $MODEL --atlas "$OUT/full/atlas.pt" --data "$DATA" --desc "$DESC" --out "$OUT/export" --res-per-bin 8 > "$OUT/export.log" 2>&1 \
  || fin FAILED "export (export.log); atlas.pt is complete" 9
tail -3 "$OUT/export.log"
rm -rf "$OUT/full/ckpt"   # the complete atlas.pt supersedes its checkpoints
fin DONE "atlas $(du -h "$OUT/full/atlas.pt" | cut -f1), bundle $(du -h "$OUT/export/firstread_$TAG.pt" | cut -f1)"
