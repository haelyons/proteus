#!/bin/bash
# Scaling benchmark. Env: JOB OUT FS DEADLINE; see specs/scale.md.
set -uo pipefail
: "${JOB:?}" "${OUT:?}" "${FS:?}" "${DEADLINE:?}"
PYTHON=${PYTHON:-python3}
SMOKE=${SMOKE:-0}
SCRATCH=${SCRATCH:-$HOME/scratch/$JOB}
NS=${NS:-"1 2 4 8"}
REF_RUN_JSON=$("$PYTHON" - "${REF_RUN_JSON:-$(dirname "$0")/ref_run.json}" <<'PYREF'
import sys
from pathlib import Path
print(Path(sys.argv[1]).resolve())
PYREF
) || exit 2
KEEP=${KEEP:-0}
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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
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

stage_cache() {
  mkdir -p "$SCRATCH/hf" || return
}
preflight() {
  "$PYTHON" -m pip install --user -q -U 'pillow>=10' 'transformers>=4.46' pydssp || return
  "$PYTHON" -c 'import torch, triton, transformers, PIL, pydssp; from transformers import EsmModel; print("preflight ok")'
}
host_record() {
  mkdir -p "$OUT/host"
  "$PYTHON" - "$SCRATCH" "$FS" <<'PY'
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
import torch

def capture(args):
    if not shutil.which(args[0]):
        return None
    p = subprocess.run(args, capture_output=True, text=True)
    return {'rc': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
versions = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()}
for name in ('triton', 'transformers'):
    try:
        versions[name] = importlib.import_module(name).__version__
    except ImportError:
        versions[name] = None
host = {'kernel': platform.uname()._asdict(), 'versions': versions,
        'omp_threads': int(os.environ['OMP_NUM_THREADS']),
        'lscpu': capture(['lscpu']), 'numactl': capture(['numactl', '-H']),
        'free': capture(['free', '-b']), 'df': capture(['df', '-B1', *sys.argv[1:]]),
        'fs_mount': capture(['findmnt', '-T', sys.argv[2], '-o', 'TARGET,SOURCE,FSTYPE,OPTIONS'])}
(Path(os.environ['OUT']) / 'host' / 'host.json').write_text(json.dumps(host, indent=1))
PY
}
gpu_record() {
  nvidia-smi --query-gpu=index,uuid,serial,name,pci.bus_id,driver_version,vbios_version,memory.total,power.limit,power.default_limit,clocks.max.sm,ecc.mode.current,compute_mode,temperature.gpu --format=csv > "$OUT/host/gpus.csv" || return
  nvidia-smi -q > "$OUT/host/nvidia_smi_q.txt"
}
shards() {
  local n=$1 i pid bad=0
  shift
  local -a pids=()
  trap 'for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done; exit 143' TERM INT
  for ((i=0; i<n; i++)); do
    step "shard_n${n}_s${i}" env CUDA_VISIBLE_DEVICES="$i" "$PYTHON" scripts/run_atlas.py "$@" --pilot-from "$SCRATCH/pilot.pt" --shard "$i/$n" --out "$SCRATCH/n$n/s$i" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || bad=1; done
  trap - TERM INT
  return "$bad"
}
archive() {
  "$PYTHON" - "$1" "$2" "${3:-}" <<'PY'
import json
import shutil
import sys
from pathlib import Path
src, dst = map(Path, sys.argv[1:3])
dst.mkdir(parents=True, exist_ok=True)
for pattern in ('*.json', '*.jsonl', '*.log', '*.txt'):
    for p in src.glob(pattern):
        shutil.copy2(p, dst / p.name)
sizes = {p.name: p.stat().st_size for p in src.glob('*.pt')}
(dst / 'output_files.json').write_text(json.dumps(sizes))
if sys.argv[3] and Path(sys.argv[3]).is_file():
    shutil.copy2(sys.argv[3], dst / 'run.log')
PY
}
digest_runs() {
  local count=$1 i pid bad=0
  shift
  local -a pids=()
  trap 'for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done; exit 143' TERM INT
  for ((i=0; i<count; i++)); do
    step "digest_$i" env CUDA_VISIBLE_DEVICES="$i" "$PYTHON" scripts/batch_digest.py "$@" --out "$OUT/digest_$i.json" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || bad=1; done
  trap - TERM INT
  return "$bad"
}
digest_compare() {
  "$PYTHON" - "$1" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path
out = Path(os.environ['OUT'])
pairs = []
for i in range(1, int(sys.argv[1])):
    p = subprocess.run([os.environ['PYTHON'], 'scripts/batch_digest.py', '--compare',
                        str(out / 'digest_0.json'), str(out / f'digest_{i}.json')], capture_output=True, text=True)
    pairs.append({'gpu': i, 'equal': p.returncode == 0,
                  'first_diff': None if p.returncode == 0 else (p.stdout + p.stderr).strip()})
equal = all(p['equal'] for p in pairs)
(out / 'digest_compare.json').write_text(json.dumps({'all_equal': equal, 'pairs': pairs}, indent=1))
sys.exit(0 if equal else 1)
PY
}
naive_pair() {
  local pair=$1 count=$2 offset=$3 gpus=$4 pid bad=0
  shift 4
  local -a bounds=() pids=() pilot=()
  if [ "$pair" = prefix ]; then
    bounds=(--max-batches "$count")
  else
    bounds=(--plan-offset "$offset" --stop-after "$count")
    pilot=(--pilot-from "$SCRATCH/pilot.pt")
  fi
  trap 'for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done; exit 143' TERM INT
  step "${pair}_filtered" env CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/run_atlas.py "$@" "${bounds[@]}" "${pilot[@]}" --out "$SCRATCH/naive/$pair/filtered" &
  pids+=("$!")
  if [ "$gpus" -lt 2 ] && [ "$SMOKE" != 1 ]; then
    wait "${pids[0]}" || bad=1
    pids=()
  fi
  step "${pair}_naive" env CUDA_VISIBLE_DEVICES="$((gpus > 1 ? 1 : 0))" "$PYTHON" scripts/run_atlas.py "$@" "${bounds[@]}" --naive --out "$SCRATCH/naive/$pair/naive" &
  pids+=("$!")
  for pid in "${pids[@]}"; do wait "$pid" || bad=1; done
  trap - TERM INT
  return "$bad"
}
export -f record step stage_cache preflight host_record gpu_record shards archive digest_runs digest_compare naive_pair
export FS

if [ "$SMOKE" = 1 ]; then
  export HF_HUB_OFFLINE=1
  MODEL=8M; DEV=cpu; DT=fp32; BACKEND=torch; TPB=4096; PT=8192; K=16; KR=4
  NS="1 2"; GPUS=1
  DATA=${SMOKE_DATA:-data/swissprot_test500.pt}
  HELD=lists/heldout_created_after_2021-11-17.tsv
  LIMITS=(--max-seqs 128)
  EDGES=(0.3 0.5 0.7 0.9 0.99)
  NAIVE_PREFIX=2; NAIVE_OFFSET=4; NAIVE_STEADY=1
else
  unset HF_HUB_OFFLINE
  MAX_N=1
  for n in $NS; do
    [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "invalid NS"; exit 2; }
    if [ "$n" -gt "$MAX_N" ]; then MAX_N=$n; fi
  done
  THREADS=$(( $(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc) / MAX_N ))
  if [ "$THREADS" -lt 1 ]; then THREADS=1; fi
  export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
  export HF_HOME=$SCRATCH/hf HF_HUB_CACHE=$SCRATCH/hf/hub
  MODEL=${MODEL:-650M}; DEV=cuda; DT=bf16; BACKEND=triton
  TPB=${TPB:-32768}; PT=${PT:-262144}; K=${K:-64}; KR=${KR:-64}
  DATA=$FS/cache/swissprot_2026_03.pt
  HELD=heldout_created_after_2021-11-17.tsv
  LIMITS=(); EDGES=(0.5 0.9 0.99 0.999 0.9999 0.99999)
  NAIVE_PREFIX=${NAIVE_PREFIX:-60}; NAIVE_STEADY=${NAIVE_STEADY:-40}; NAIVE_OFFSET=${NAIVE_OFFSET:-4500}
  run_step stage_cache stage_cache
  run_step preflight preflight
  # Download the weights once, outside every timed GPU step (the first load would otherwise land in the digest or pilot step).
  run_step model_download "$PYTHON" -c "import sys; sys.path.insert(0, '.'); from atlas.esm import load; load('$MODEL', device='cpu'); print('model cached')"
  GPU_IDS=$(nvidia-smi --query-gpu=index --format=csv,noheader) || exit 3
  GPUS=$(printf '%s\n' "$GPU_IDS" | wc -l)
fi
for n in $NS; do [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "invalid NS"; exit 2; }; done
COMMON=(--model "$MODEL" --data "$DATA" --device "$DEV" --dtype "$DT" --backend "$BACKEND"
        --tokens-per-batch "$TPB" --pilot-tokens "$PT" --K "$K" --Kr "$KR" --edge-q "${EDGES[@]}"
        --seed-thresholds --seed 0 --exclude-acc "$HELD" --prefetch 2 --stats-impl v3 --merge-impl stacked
        --gram-flush-every 1000 --ckpt-every 1000000 --deadline "$DEADLINE" "${LIMITS[@]}")
RUN=(env CUDA_VISIBLE_DEVICES=0 "$PYTHON")
run_step host host_record
if [ "$SMOKE" = 1 ]; then
  for name in gpu_record smi_logger gpu_tests stats_gpu_tests cross_gpu_digest digest_compare; do skip "$name" 'SMOKE: CPU only'; done
else
  run_step gpu_record gpu_record
  Q=timestamp,index,uuid,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,utilization.memory,memory.used
  echo "$Q" > "$OUT/smi.csv"
  nvidia-smi --query-gpu="$Q" --format=csv,noheader,nounits -l 1 >> "$OUT/smi.csv" 2> "$OUT/smi.err" &
  SMI_PID=$!
  ( while kill -0 "$SMI_PID" 2>/dev/null; do ps -o pid,ppid,etime,stat,cmd -p "$SMI_PID"; sleep 30; done ) > "$OUT/ps.log" 2>&1 &
  PS_PID=$!
  run_step smi_logger kill -0 "$SMI_PID"
  run_step gpu_tests env CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tests.test_triton_kernel --cuda
  run_step stats_gpu_tests env CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tests.test_stats_triton --cuda
  read -r -a DIGEST <<< "${DIGEST_BATCHES:-0 1000}"
  run_step cross_gpu_digest digest_runs "$GPUS" --model "$MODEL" --data "$DATA" --device cuda --dtype "$DT" \
    --batch "${DIGEST[@]}" --tokens-per-batch "$TPB" --seed 0 --exclude-acc "$HELD"
  run_step digest_compare digest_compare "$GPUS"
fi
run_step pilot "${RUN[@]}" scripts/run_atlas.py "${COMMON[@]}" --out "$SCRATCH/pilot" --pilot-out "$SCRATCH/pilot.pt" --pilot-only
run_step single "${RUN[@]}" scripts/run_atlas.py "${COMMON[@]}" --pilot-from "$SCRATCH/pilot.pt" --out "$SCRATCH/single"
run_step archive_single archive "$SCRATCH/single" "$OUT/single" "$OUT/single.log"
if [ "$SMOKE" = 1 ]; then
  skip single_vs_ref 'SMOKE: different workload from reference'
else
  run_step single_vs_ref "$PYTHON" scripts/compare_atlas.py "$SCRATCH/single/atlas.pt" --digests "$REF_RUN_JSON" --out "$OUT/single_vs_ref.json"
fi
LARGEST=0
for n in $NS; do
  [ "$n" -gt 1 ] || continue
  if [ "$SMOKE" != 1 ] && [ "$n" -gt "$GPUS" ]; then skip "shards_n$n" 'exceeds visible GPU count'; continue; fi
  mkdir -p "$OUT/n$n"
  run_step "shards_n$n" shards "$n" "${COMMON[@]}"
  SHARDS=()
  for ((i=0; i<n; i++)); do SHARDS+=("$SCRATCH/n$n/s$i/shard.pt"); done
  TIME=()
  if [ -x /usr/bin/time ]; then TIME=(/usr/bin/time -v -o "$OUT/n$n/merge_time.txt"); fi
  run_step "merge_n$n" "${TIME[@]}" "$PYTHON" scripts/merge_atlas.py \
    --out "$SCRATCH/n$n/merged" "${SHARDS[@]}" --device cpu
  "$PYTHON" - "$OUT/timing/step_merge_n$n.json" "${#TIME[@]}" <<'PYTIME'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
s = json.loads(p.read_text())
s['time_v'] = int(sys.argv[2]) > 0
p.write_text(json.dumps(s))
PYTIME
  run_step "compare_n$n" "$PYTHON" scripts/compare_atlas.py "$SCRATCH/n$n/merged/atlas.pt" "$SCRATCH/single/atlas.pt" --rtol 1e-5 --out "$OUT/n$n/gate_compare.json"
  if [ "$SMOKE" = 1 ]; then
    skip "vs_ref_n$n" 'SMOKE: different workload from reference'
  else
    run_step "vs_ref_n$n" "$PYTHON" scripts/compare_atlas.py "$SCRATCH/n$n/merged/atlas.pt" --digests "$REF_RUN_JSON" --out "$OUT/n$n/vs_ref.json"
  fi
  for ((i=0; i<n; i++)); do
    run_step "archive_n${n}_s$i" archive "$SCRATCH/n$n/s$i" "$OUT/n$n/s$i" "$OUT/shard_n${n}_s$i.log"
  done
  run_step "archive_n${n}_merged" archive "$SCRATCH/n$n/merged" "$OUT/n$n/merged" "$OUT/merge_n$n.log"
  # Only delete the explicit shard outputs created by this job.
  run_step "delete_shards_n$n" rm -f -- "${SHARDS[@]}"
  if [ -f "$SCRATCH/n$n/merged/atlas.pt" ] && [ "$n" -gt "$LARGEST" ]; then LARGEST=$n; fi
  if [ "$SMOKE" != 1 ]; then run_step "smi_alive_n$n" kill -0 "$SMI_PID"; fi
done
mkdir -p "$OUT/naive"
for pair in prefix steady; do
  if [ "$pair" = prefix ]; then COUNT=$NAIVE_PREFIX; else COUNT=$NAIVE_STEADY; fi
  run_step "naive_$pair" naive_pair "$pair" "$COUNT" "$NAIVE_OFFSET" "$GPUS" "${COMMON[@]}"
  for mode in filtered naive; do
    run_step "archive_${pair}_$mode" archive "$SCRATCH/naive/$pair/$mode" "$OUT/naive/$pair/$mode" "$OUT/${pair}_$mode.log"
  done
  if [ "$pair" = prefix ]; then
    run_step cmp_prefix "$PYTHON" scripts/compare_atlas.py "$SCRATCH/naive/prefix/filtered/atlas.pt" "$SCRATCH/naive/prefix/naive/atlas.pt" --out "$OUT/naive/cmp_prefix.json"
  fi
done
if [ "$KEEP" = 1 ]; then
  run_step keep_single cp "$SCRATCH/single/atlas.pt" "$OUT/single/atlas.pt"
  if [ "$LARGEST" -gt 1 ]; then run_step keep_merged cp "$SCRATCH/n$LARGEST/merged/atlas.pt" "$OUT/n$LARGEST/merged/atlas.pt"; fi
fi
if [ "$SMOKE" != 1 ]; then run_step smi_alive_end kill -0 "$SMI_PID"; fi
timeline
run_step scaling_report "$PYTHON" scripts/scaling_report.py "$OUT"
exit "$FAILED"
