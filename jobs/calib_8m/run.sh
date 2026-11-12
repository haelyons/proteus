#!/bin/bash
# Frozen 8M calibration. Env: JOB OUT FS DEADLINE; see specs/calib.md.
set -uo pipefail
: "${JOB:?}" "${OUT:?}" "${FS:?}" "${DEADLINE:?}"
PYTHON=${PYTHON:-python3}
SMOKE=${SMOKE:-0}
if [ -z "${SCRATCH:-}" ]; then  # default: local disk under HOME; its parent does not exist on a fresh instance
  SCRATCH=$HOME/scratch/$JOB; mkdir -p "$HOME/scratch" || exit 2
fi
WORKERS=${WORKERS:-$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))}
PER_LAYER=${PER_LAYER:-250}
if [ "$SMOKE" = 1 ]; then
  PER_LAYER=4
fi
[[ "$WORKERS" =~ ^[1-9][0-9]*$ && "$PER_LAYER" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$DEADLINE" =~ ^[0-9]+$ ]] || { echo "invalid DEADLINE"; exit 2; }
# Require fresh OUT and reject symlinks that could redirect writes.
PATHS=$("$PYTHON" - "$OUT" "$SCRATCH" "$FS" <<'PY'
import sys
from pathlib import Path
out, scratch, fs = [Path(p).absolute() for p in sys.argv[1:]]
RUNNER_FILES = {'started.json', 'job.log'}  # written into OUT by infra/lambda_job.py's inbox runner before run.sh starts
if out.is_symlink() or (out.exists() and (not out.is_dir() or any(p.name not in RUNNER_FILES for p in out.iterdir()))):
    sys.exit("OUT must be a new or empty directory, not a symlink")
if scratch.is_symlink() or (scratch.exists() and not scratch.is_dir()):
    sys.exit("SCRATCH must be a directory, not a symlink")
for p in (out, scratch):
    if not p.parent.is_dir():
        sys.exit("OUT and SCRATCH parents must exist")
out, scratch, fs = out.resolve(), scratch.resolve(), fs.resolve()
if out == scratch or out in scratch.parents or scratch in out.parents:
    sys.exit("OUT and SCRATCH must be separate directories")
if scratch == fs or fs in scratch.parents:
    sys.exit("SCRATCH must not be on FS")
print(out)
print(scratch)
PY
) || exit 2
OUT=${PATHS%%$'\n'*}
SCRATCH=${PATHS#*$'\n'}
mkdir -p "$OUT/timing" "$SCRATCH" || exit 2
export OUT SCRATCH JOB PYTHON SMOKE DEADLINE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}
export TMPDIR="$SCRATCH/tmp" XDG_CACHE_HOME="$SCRATCH/cache" TRITON_CACHE_DIR="$SCRATCH/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/cache/inductor" PIP_CACHE_DIR="$SCRATCH/cache/pip"
export PYTHONUSERBASE="$SCRATCH/python" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR"
SMI_PID=""; PS_PID=""; PIDS=(); FAILED=0

timeline() {
  "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path
out = Path(os.environ['OUT'])
steps = [json.loads(p.read_text()) for p in (out / 'timing').glob('step_*.json')]
(out / 'timeline.json').write_text(json.dumps({'job': os.environ['JOB'],
    'steps': sorted(steps, key=lambda s: (s['start'], s['step']))}, indent=1))
PY
}
cleanup() {
  rc=$?
  trap - EXIT
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "$SMI_PID" "$PS_PID"; do
    if [ -n "$pid" ]; then kill "$pid" 2>/dev/null || true; fi
  done
  for pid in "${PIDS[@]}" "$SMI_PID" "$PS_PID"; do
    if [ -n "$pid" ]; then wait "$pid" 2>/dev/null || true; fi
  done
  timeline
  rm -rf -- "$TMPDIR"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

record() {
  "$PYTHON" - "$@" <<'PY'
import json
import os
import sys
from pathlib import Path
name, start, end, rc, *reason = sys.argv[1:]
s = {'step': name, 'start': float(start), 'end': float(end), 'rc': int(rc)}
if reason:
    s.update(skipped=True, reason=reason[0])
(Path(os.environ['OUT']) / 'timing' / ('step_' + name + '.json')).write_text(json.dumps(s))
PY
}
skip() {
  local now
  now=$(date +%s.%N)
  echo "[atlas] $1 skipped: $2" | tee "$OUT/$1.log"
  record "$1" "$now" "$now" 0 "$2"
}
step() {
  local name=$1 start end rc left
  shift
  start=$(date +%s.%N)
  left=$(( DEADLINE - $(date +%s) ))
  echo "[atlas] $name start (${left}s left)"
  rc=0
  if [ "$left" -le 0 ]; then
    echo "deadline exceeded" > "$OUT/$name.log"
    rc=124
  else
    timeout --kill-after=10 "$left" bash -c '"$@"' _ "$@" > "$OUT/$name.log" 2>&1 || rc=$?
  fi
  end=$(date +%s.%N)
  record "$name" "$start" "$end" "$rc"
  echo "[atlas] $name rc=$rc"
  return "$rc"
}
run_step() { step "$@" || FAILED=1; }

preflight() {
  "$PYTHON" -m pip install --user -q -U 'pillow>=10' 'transformers>=4.46' pydssp || return
  "$PYTHON" -c 'import torch, triton, transformers, PIL, pydssp; from transformers import EsmModel; print("preflight ok")'
}
gpu_record() {
  nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit,clocks.max.sm --format=csv,noheader > "$OUT/gpus.csv" || return
  nvidia-smi -q > "$OUT/nvidia_smi_q.txt"
}
smoke_descriptors() {
  "$PYTHON" - <<'PY'
import gzip
import os
from pathlib import Path
import torch
root = Path(os.environ['OUT']) / 'smoke_inputs'
root.mkdir()
data = torch.load(os.environ['DATA'], map_location='cpu', weights_only=False)
# Whole-build descriptors retain the test set's original sequence indices.
acc = data['acc']
lengths = [min(len(seq), 1022) for seq in data['seq']]
offset = torch.tensor([0] + lengths).cumsum(0)
n = int(offset[-1])
g = torch.Generator().manual_seed(0)
d = {'acc': acc, 'seq_idx': torch.arange(len(acc)), 'offset': offset,
     'ss3': torch.randint(3, (n,), generator=g), 'plddt': torch.full((n,), 90.),
     'burial': torch.randint(46, (n,), generator=g).float(),
     'phi': torch.rand(n, generator=g) * 360 - 180, 'psi': torch.rand(n, generator=g) * 360 - 180}
torch.save(d, root / 'descriptors.pt')
with gzip.open(root / 'uniref50.tsv.gz', 'wt') as f:
    f.write('Entry\tUniRef50\n')
    for a in acc:
        f.write(f'{a}\tUniRef50_{a}\n')
PY
}
archive() {
  local name=$1
  mkdir -p "$OUT/$name" || return
  cp "$SCRATCH/$name/run.json" "$SCRATCH/$name/batches.jsonl" "$OUT/$name/"
}
score_source() {
  if [ "$SMOKE" != 1 ]; then
    "$PYTHON" scripts/compare_sources.py score "$@"
    return
  fi
  "$PYTHON" - "$@" <<'PY'
import runpy
import sys
from atlas.ruler import LEVELS
# The frozen smoke quantiles yield five bins; the ruler expects six.
# Keep the stored atlas unchanged and score its last bin as the peak.
LEVELS['peak'] = 4
sys.argv = ['scripts/compare_sources.py', 'score', *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name='__main__')
PY
}
export -f timeline record skip step preflight gpu_record smoke_descriptors archive score_source

if [ "$SMOKE" = 1 ]; then
  DEV=cpu; DT=fp32; BACKEND=torch; TPB=4096; PT=8192; K=16; KR=8
  DATA=${SMOKE_DATA:-data/swissprot_test500.pt}
  HELD=lists/heldout_created_after_2021-11-17.tsv
  DESC="$OUT/smoke_inputs/descriptors.pt"
  UNIREF="$OUT/smoke_inputs/uniref50.tsv.gz"
  LIMITS=(--max-seqs 64); EDGES=(0.3 0.5 0.7 0.9 0.99)
  export DATA
  step smoke_descriptors smoke_descriptors || exit 1
  for name in preflight gpu_record smi_logger gpu_tests stats_gpu_tests; do skip "$name" 'SMOKE: CPU only'; done
else
  export HF_HOME=$FS/cache/hf
  DEV=cuda; DT=bf16; BACKEND=triton; TPB=32768; PT=262144; K=64; KR=256
  DATA=$FS/cache/swissprot_2026_03.pt
  HELD=heldout_created_after_2021-11-17.tsv
  DESC=$FS/descriptors_swissprot_2026_03.pt
  UNIREF=lists/swissprot_2026_03_uniref50.tsv.gz
  LIMITS=(); EDGES=(0.5 0.9 0.99 0.999 0.9999 0.99999)
  step preflight preflight || exit 2
  step gpu_record gpu_record || exit 3
  Q=timestamp,power.draw,power.limit,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used
  echo "$Q" > "$OUT/smi.csv"
  nvidia-smi --query-gpu="$Q" --format=csv,noheader,nounits -l 1 >> "$OUT/smi.csv" 2> "$OUT/smi.err" &
  SMI_PID=$!
  ( while kill -0 "$SMI_PID" 2>/dev/null; do ps -o pid,ppid,etime,stat,cmd -p "$SMI_PID"; sleep 30; done ) > "$OUT/ps.log" 2>&1 &
  PS_PID=$!
  step smi_logger kill -0 "$SMI_PID" || exit 3
  step gpu_tests "$PYTHON" -m tests.test_triton_kernel --cuda || exit 5
  step stats_gpu_tests "$PYTHON" -m tests.test_stats_triton --cuda || exit 5
fi
COMMON=(--model 8M --data "$DATA" --device "$DEV" --dtype "$DT" --backend "$BACKEND"
        --tokens-per-batch "$TPB" --K "$K" --Kr "$KR" --edge-q "${EDGES[@]}"
        --exclude-acc "$HELD" --pilot-tokens "$PT" --seed-thresholds --ckpt-every 2000
        --deadline "$DEADLINE" "${LIMITS[@]}")
for name in trained shuf1 shuf2 shuf3; do
  CONTROL=(); COLUMNS=neurons_pre,neurons_post
  if [ "$name" = trained ]; then
    COLUMNS+=,randdir:512:0
  else
    CONTROL=(--control shuffled --control-seed "${name#shuf}")
  fi
  step "$name" "$PYTHON" scripts/run_atlas.py "${COMMON[@]}" "${CONTROL[@]}" \
    --columns "$COLUMNS" --source-cap neurons_post=0.1 --out "$SCRATCH/$name" || exit 1
  step "archive_$name" archive "$name" || exit 1
done
for name in trained shuf1 shuf2 shuf3; do
  if [ "$SMOKE" = 1 ]; then
    CONTROL=()
    if [ "$name" != trained ]; then CONTROL=(--control shuffled --control-seed "${name#shuf}"); fi
    # The legacy single-source path has no --columns argument.
    step "reference_$name" "$PYTHON" scripts/run_atlas.py "${COMMON[@]}" "${CONTROL[@]}" \
      --out "$SCRATCH/reference_$name" || exit 1
    REF="$SCRATCH/reference_$name/atlas.pt"
  else
    case "$name" in
      trained) REF="$FS/runs/atlas-batch-20261105-0258/a8_pre/atlas.pt";;
      shuf1) REF="$FS/runs/atlas-batch2-20261105-0348/a8_shuf1/atlas.pt";;
      shuf2) REF="$FS/runs/atlas-batch-20261105-0258/a8_shuf2/atlas.pt";;
      shuf3) REF="$FS/runs/atlas-batch2-20261105-0348/a8_shuf3/atlas.pt";;
    esac
  fi
  # A recorded mismatch is evidence, not a reason to stop calibration.
  step "repro_$name" "$PYTHON" scripts/compare_atlas.py "$SCRATCH/$name/atlas_neurons_pre.pt" \
    "$REF" --out "$OUT/repro_$name.json" || {
    [ -f "$OUT/repro_$name.json" ] || FAILED=1
  }
done
mkdir -p "$OUT/planted" "$OUT/pooling" "$OUT/atlases" || exit 1
ATLASES=()
for name in trained shuf1 shuf2 shuf3; do ATLASES+=("$name=$SCRATCH/$name/atlas_neurons_post.pt"); done
step pooling_sample "$PYTHON" scripts/compare_sources.py sample --atlas "${ATLASES[@]}" \
  --per-layer "$PER_LAYER" --seed 0 --out "$OUT/pooling/sample.json" || exit 1
for name in trained shuf1 shuf2 shuf3; do
  step "pooling_score_$name" score_source --sample "$OUT/pooling/sample.json" \
    --source "$name" --atlas "$SCRATCH/$name/atlas_neurons_post.pt" --desc "$DESC" --data "$DATA" \
    --uniref "$UNIREF" --workers "$WORKERS" --out "$OUT/pooling/scores" || exit 1
done
step calibration "$PYTHON" scripts/separation_gate.py --checks "$OUT" --out "$OUT/calibration.json" || exit 1
step copy_atlases cp "$SCRATCH/trained/atlas_neurons_post.pt" "$SCRATCH/trained/atlas_randdir_512_0.pt" "$OUT/atlases/" || exit 1
step report "$PYTHON" scripts/calib_report.py "$OUT" || exit 1
if [ "$SMOKE" != 1 ]; then run_step smi_alive_end kill -0 "$SMI_PID"; fi
exit "$FAILED"
