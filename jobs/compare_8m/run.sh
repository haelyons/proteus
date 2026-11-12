#!/bin/bash
# Frozen 8M comparison. Env: JOB OUT FS DEADLINE; see specs/compare_8m.md.
set -uo pipefail
: "${JOB:?}" "${OUT:?}" "${FS:?}" "${DEADLINE:?}"
PYTHON=${PYTHON:-python3}
SMOKE=${SMOKE:-0}
CALIB=${CALIB:-$FS/runs/calib-8m-20261112-1018}
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
inputs() {
  "$PYTHON" - <<'PYINPUTS'
import json
import os
from pathlib import Path
from scripts.compare_sources import sha256
calib = Path(os.environ['CALIB'])
paths = [calib / 'atlases' / name for name in ('atlas_neurons_post.pt', 'atlas_randdir_512_0.pt')]
paths += [calib / 'pooling/sample.json']
paths += [calib / 'pooling/scores' / f'{name}.jsonl' for name in ('trained', 'shuf1', 'shuf2', 'shuf3')]
paths += [Path(os.environ[key]) for key in ('DATA', 'DESC', 'UNIREF', 'HELD')]
result = []
for p in paths:
    if not p.exists():
        raise FileNotFoundError(p)
    size = p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.glob('part_*.pt'))
    result.append({'path': str(p), 'size': size, 'sha256': sha256(p)})
(Path(os.environ['OUT']) / 'inputs.json').write_text(json.dumps(result, indent=2) + '\n')
PYINPUTS
}
repro() {
  "$PYTHON" - <<'PYREPRO'
import json
import os
from pathlib import Path
from scripts.separation_gate import records
out, calib = Path(os.environ['OUT']), Path(os.environ['CALIB'])
result = {}
for name, reference in (('N', 'trained'), ('S1', 'shuf1'), ('S2', 'shuf2'), ('S3', 'shuf3')):
    current = {(r['layer'], r['col']): r['score_sha256'] for r in records(out / 'scores' / f'{name}.jsonl')}
    previous = {(r['layer'], r['col']): r['score_sha256'] for r in records(calib / 'pooling/scores' / f'{reference}.jsonl')}
    keys = current.keys() | previous.keys()
    result[name] = {'n': len(keys), 'identical': sum(k in current and k in previous and current[k] == previous[k] for k in keys)}
(out / 'repro.json').write_text(json.dumps(result, indent=2) + '\n')
PYREPRO
}
copy_atlases() {
  "$PYTHON" - <<'PYATLASES'
import json
import os
import shutil
from pathlib import Path
from scripts.compare_sources import sha256
root = Path(os.environ['OUT']) / 'atlases'
root.mkdir()
digests = {}
for c in (1, 2, 3):
    dest = root / f'RS{c}.pt'
    shutil.copyfile(Path(os.environ['SCRATCH']) / f'shuf{c}' / 'atlas_randdir_512_0.pt', dest)
    digests[f'RS{c}'] = sha256(dest)
(root / 'digests.json').write_text(json.dumps(digests, indent=2) + '\n')
PYATLASES
}
export -f timeline record skip step preflight gpu_record smoke_descriptors archive inputs repro copy_atlases

if [ "$SMOKE" = 1 ]; then
  DEV=cpu; DT=fp32; BACKEND=torch; TPB=4096; PT=8192; K=16; KR=16
  DATA=${SMOKE_DATA:-data/swissprot_test500.pt}
  HELD=lists/heldout_created_after_2021-11-17.tsv
  DESC="$OUT/smoke_inputs/descriptors.pt"
  UNIREF="$OUT/smoke_inputs/uniref50.tsv.gz"
  LIMITS=(--max-seqs 32); EDGES=(0.3 0.5 0.7 0.9 0.95 0.99)
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
if [ "$SMOKE" = 1 ]; then
  CALIB="$OUT/smoke_inputs"
  mkdir -p "$CALIB/atlases" "$CALIB/pooling" || exit 1
  for name in trained shuf1 shuf2 shuf3; do
    CONTROL=(); COLUMNS=neurons_pre,neurons_post
    if [ "$name" = trained ]; then
      COLUMNS=neurons_post,randdir:512:0
    else
      CONTROL=(--control shuffled --control-seed "${name#shuf}")
    fi
    step "smoke_$name" "$PYTHON" scripts/run_atlas.py "${COMMON[@]}" "${CONTROL[@]}" \
      --columns "$COLUMNS" --source-cap neurons_post=0.1 --out "$SCRATCH/smoke_$name" || exit 1
  done
  step smoke_copy_atlases cp "$SCRATCH/smoke_trained/atlas_neurons_post.pt" \
    "$SCRATCH/smoke_trained/atlas_randdir_512_0.pt" "$CALIB/atlases/" || exit 1
  ATLASES=()
  for name in trained shuf1 shuf2 shuf3; do ATLASES+=("$name=$SCRATCH/smoke_$name/atlas_neurons_post.pt"); done
  step smoke_sample "$PYTHON" scripts/compare_sources.py sample --atlas "${ATLASES[@]}" \
    --per-layer "$PER_LAYER" --seed 0 --out "$CALIB/pooling/sample.json" || exit 1
  for name in trained shuf1 shuf2 shuf3; do
    step "smoke_score_$name" "$PYTHON" scripts/compare_sources.py score --sample "$CALIB/pooling/sample.json" \
      --source "$name" --atlas "$SCRATCH/smoke_$name/atlas_neurons_post.pt" --desc "$DESC" --data "$DATA" \
      --uniref "$UNIREF" --workers "$WORKERS" --out "$CALIB/pooling/scores" || exit 1
  done
fi
export CALIB DATA DESC UNIREF HELD
step inputs inputs || exit 1
for c in 1 2 3; do
  step "shuf$c" "$PYTHON" scripts/run_atlas.py "${COMMON[@]}" --control shuffled --control-seed "$c" \
    --columns neurons_post,randdir:512:0 --source-cap neurons_post=0.1 --out "$SCRATCH/shuf$c" || exit 1
  step "archive_shuf$c" archive "shuf$c" || exit 1
done
N="$CALIB/atlases/atlas_neurons_post.pt"
R="$CALIB/atlases/atlas_randdir_512_0.pt"
NAMES=(N R S1 S2 S3 RS1 RS2 RS3)
PATHS=("$N" "$R")
for c in 1 2 3; do PATHS+=("$SCRATCH/shuf$c/atlas_neurons_post.pt"); done
for c in 1 2 3; do PATHS+=("$SCRATCH/shuf$c/atlas_randdir_512_0.pt"); done
ATLASES=()
for i in "${!NAMES[@]}"; do ATLASES+=("${NAMES[$i]}=${PATHS[$i]}"); done
step sample "$PYTHON" scripts/compare_sources.py sample --atlas "${ATLASES[@]}" \
  --per-layer "$PER_LAYER" --seed 0 --out "$OUT/sample.json" || exit 1
for i in "${!NAMES[@]}"; do
  step "score_${NAMES[$i]}" "$PYTHON" scripts/compare_sources.py score --sample "$OUT/sample.json" \
    --source "${NAMES[$i]}" --atlas "${PATHS[$i]}" --desc "$DESC" --data "$DATA" \
    --uniref "$UNIREF" --workers "$WORKERS" --out "$OUT/scores" || exit 1
done
step repro repro || exit 1
step summary "$PYTHON" scripts/compare_sources.py summarise --scores "$OUT/scores" \
  --pool N,R,S1,S2,S3,RS1,RS2,RS3 --out "$OUT/summary.json" || exit 1
step report "$PYTHON" scripts/compare_report.py --scores "$OUT/scores" --summary "$OUT/summary.json" \
  --repro "$OUT/repro.json" --out "$OUT/compare.json" || exit 1
step copy_atlases copy_atlases || exit 1
if [ "$SMOKE" != 1 ]; then run_step smi_alive_end kill -0 "$SMI_PID"; fi
exit "$FAILED"
