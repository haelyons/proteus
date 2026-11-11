#!/bin/bash
# Shared-pilot sharding gate. Env: JOB OUT FS DEADLINE; optional NSHARDS, SMOKE, SMOKE_DATA.
set -euo pipefail
: "${JOB:?}" "${OUT:?}" "${FS:?}" "${DEADLINE:?}"
SMOKE=${SMOKE:-0}
NSHARDS=${NSHARDS:-2}
PYTHON=${PYTHON:-python3}
[[ "$NSHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid NSHARDS"; exit 2; }
[[ "$DEADLINE" =~ ^[0-9]+$ ]] || { echo "invalid DEADLINE"; exit 2; }

# Require a fresh output directory: existing symlinks must never redirect our writes.
OUT=$("$PYTHON" - "$OUT" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1]).absolute()
if p.is_symlink() or (p.exists() and (not p.is_dir() or any(p.iterdir()))):
    sys.exit("OUT must be a new or empty directory, not a symlink")
if not p.parent.is_dir():
    sys.exit("OUT parent must exist")
print(p.resolve())
PY
)
mkdir -p "$OUT"
export OUT JOB
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}
export TMPDIR="$OUT/tmp" XDG_CACHE_HOME="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$OUT/cache/inductor" PIP_CACHE_DIR="$OUT/cache/pip"
export PYTHONUSERBASE="$OUT/python" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$OUT/timing"
SMI_PID=""
PIDS=()

timeline() {
  "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path
out = Path(os.environ["OUT"])
steps = [json.loads(p.read_text()) for p in (out / "timing").glob("step_*.json")]
(out / "timeline.json").write_text(json.dumps({"job": os.environ["JOB"],
                                            "steps": sorted(steps, key=lambda s: (s["start"], s["step"]))}, indent=1))
PY
}
cleanup() {
  rc=$?
  trap - EXIT
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  if [ -n "$SMI_PID" ]; then kill "$SMI_PID" 2>/dev/null || true; fi
  wait 2>/dev/null || true
  timeline
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
timeline

step() {
  local name=$1 start end rc left
  shift
  start=$(date +%s)
  left=$(( DEADLINE - start ))
  echo "[atlas] $name start $(date -u +%FT%TZ) (${left}s left)"
  rc=0
  if [ "$left" -le 0 ]; then
    rc=124
  else
    timeout --kill-after=10 "$left" "$@" > "$OUT/$name.log" 2>&1 || rc=$?
  fi
  end=$(date +%s)
  "$PYTHON" - "$name" "$start" "$end" "$rc" <<'PY'
import json
import os
import sys
from pathlib import Path
name, start, end, rc = sys.argv[1:]
(Path(os.environ["OUT"]) / "timing" / ("step_" + name + ".json")).write_text(
    json.dumps({"step": name, "start": int(start), "end": int(end), "rc": int(rc)}))
PY
  echo "[atlas] $name rc=$rc wall=$(( end - start ))s"
  return "$rc"
}

if [ "$SMOKE" = 1 ]; then
  MODEL=8M; DEV=cpu; DT=fp32; BACKEND=torch; TPB=4096; PT=4096
  DATA=${SMOKE_DATA:-data/swissprot_test500.pt}
  LIMITS=(--max-seqs 32 --max-batches 4)
else
  MODEL=650M; DEV=cuda; DT=bf16; BACKEND=triton; TPB=8192; PT=32768
  DATA=$FS/cache/swissprot_2026_03.pt
  LIMITS=(--max-seqs 200 --max-batches 8)
  export HF_HOME=$FS/cache/hf
  # Offline preflight, with any installs confined to OUT.
  "$PYTHON" -m pip install --no-index --user -q -U "pillow>=10" "transformers>=4.46" "pydssp" 2>&1 | tail -1
  "$PYTHON" -c "import torch, triton, transformers, PIL, pydssp; from transformers import EsmModel; print('[atlas] preflight ok')"
  nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit,clocks.max.sm --format=csv,noheader
  Q="timestamp,index,power.draw,power.limit,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used"
  echo "$Q" > "$OUT/smi.csv"
  nvidia-smi --query-gpu="$Q" --format=csv,noheader,nounits -l 1 >> "$OUT/smi.csv" 2> "$OUT/smi.err" &
  SMI_PID=$!
fi
[ -f "$DATA" ] || { echo "missing dataset: $DATA"; exit 3; }
COMMON=(--model "$MODEL" --data "$DATA" --device "$DEV" --dtype "$DT" --backend "$BACKEND"
        --tokens-per-batch "$TPB" --pilot-tokens "$PT" --K 16 --Kr 8 --seed-thresholds
        --gram-flush-every 1 --deadline "$DEADLINE" "${LIMITS[@]}")
RUN=("$PYTHON")
if [ "$SMOKE" != 1 ]; then RUN=(env CUDA_VISIBLE_DEVICES=0 "$PYTHON"); fi
step pilot "${RUN[@]}" scripts/run_atlas.py "${COMMON[@]}" --out "$OUT/pilot" --pilot-out "$OUT/pilot.pt" --pilot-only
step single "${RUN[@]}" scripts/run_atlas.py "${COMMON[@]}" --pilot-from "$OUT/pilot.pt" --out "$OUT/single"
SHARDS=()
for ((i=0; i<NSHARDS; i++)); do
  SHARDS+=("$OUT/shard_$i/shard.pt")
  if [ "$SMOKE" = 1 ]; then
    step "shard_$i" "$PYTHON" scripts/run_atlas.py "${COMMON[@]}" --pilot-from "$OUT/pilot.pt" --shard "$i/$NSHARDS" --out "$OUT/shard_$i"
  else
    step "shard_$i" env CUDA_VISIBLE_DEVICES="$i" "$PYTHON" scripts/run_atlas.py "${COMMON[@]}" --pilot-from "$OUT/pilot.pt" --shard "$i/$NSHARDS" --out "$OUT/shard_$i" &
    PIDS+=("$!")
  fi
done
failed=0
for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
PIDS=()
[ "$failed" = 0 ] || { echo "[atlas] shard failed"; exit 4; }
step merge "$PYTHON" scripts/merge_atlas.py --out "$OUT/merged" "${SHARDS[@]}" --device cpu
step compare "$PYTHON" scripts/compare_atlas.py "$OUT/merged/atlas.pt" "$OUT/single/atlas.pt" --out "$OUT/gate_compare.json"
if [ "$SMOKE" != 1 ]; then
  DIGEST=(--model "$MODEL" --data "$DATA" --device cuda --dtype "$DT" --batch 0
          --tokens-per-batch "$TPB" "${LIMITS[@]}")
  for i in 0 1; do
    step "digest_$i" env CUDA_VISIBLE_DEVICES="$i" "$PYTHON" scripts/batch_digest.py "${DIGEST[@]}" --out "$OUT/digest_$i.json"
  done
  step digest_compare "$PYTHON" scripts/batch_digest.py --compare "$OUT/digest_0.json" "$OUT/digest_1.json"
fi
echo "[atlas] shard gate passed"
