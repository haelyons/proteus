#!/bin/bash
# Frozen 650M comparison. Env: JOB OUT FS DEADLINE; see specs/compare_650m.md.
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
inputs() {
  "$PYTHON" - <<'PYINPUTS'
import json
import os
from pathlib import Path
from scripts.compare_sources import sha256
paths = [Path(os.environ[key]) for key in ('DATA', 'DESC', 'UNIREF', 'HELD')]
# The self-comparison stand-in is generated by report later in SMOKE.
if not (os.environ['SMOKE'] == '1' and os.environ['SMALL_COMPARE'] == str(Path(os.environ['OUT']) / 'compare.json')):
    paths.append(Path(os.environ['SMALL_COMPARE']))
result = []
for p in paths:
    if not p.exists():
        raise FileNotFoundError(p)
    size = p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.glob('part_*.pt'))
    result.append({'path': str(p), 'size': size, 'sha256': sha256(p)})
(Path(os.environ['OUT']) / 'inputs.json').write_text(json.dumps(result, indent=2) + '\n')
PYINPUTS
}
archive() {
  local name=$1 s
  mkdir -p "$OUT/$name" || return
  cp "$SCRATCH/$name/pilot/run.json" "$OUT/$name/pilot.json" || return
  for s in 0 1; do
    cp "$SCRATCH/$name/s$s/run.json" "$OUT/$name/s$s.run.json" || return
    cp "$SCRATCH/$name/s$s/batches.jsonl" "$OUT/$name/s$s.batches.jsonl" || return
  done
}
run_atlases() {
  local name=$1 first=$2 pid s source bad=0
  shift 2
  local -a pids=() control=()
  if [ "$name" != trained ]; then control=(--control shuffled --control-seed "${name#shuf}"); fi
  trap 'for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done; exit 143' TERM INT
  step "${name}_pilot" env CUDA_VISIBLE_DEVICES="$first" OMP_NUM_THREADS="$ATLAS_THREADS" MKL_NUM_THREADS="$ATLAS_THREADS" "$PYTHON" scripts/run_atlas.py "$@" "${control[@]}" \
    --out "$SCRATCH/$name/pilot" --pilot-out "$SCRATCH/$name/pilot.pt" --pilot-only || return 1
  for s in 0 1; do
    step "${name}_s$s" env CUDA_VISIBLE_DEVICES="$((first + s))" OMP_NUM_THREADS="$ATLAS_THREADS" MKL_NUM_THREADS="$ATLAS_THREADS" "$PYTHON" scripts/run_atlas.py "$@" "${control[@]}" \
      --pilot-from "$SCRATCH/$name/pilot.pt" --shard "$s/2" --out "$SCRATCH/$name/s$s" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || bad=1; done
  pids=()
  trap - TERM INT
  step "archive_$name" archive "$name" || bad=1
  [ "$bad" = 0 ] || return 1
  for source in neurons_post randdir_512_0; do
    step "${name}_merge_$source" env OMP_NUM_THREADS="$ATLAS_THREADS" MKL_NUM_THREADS="$ATLAS_THREADS" "$PYTHON" scripts/merge_atlas.py \
      --out "$SCRATCH/$name/merge_$source" "$SCRATCH/$name/s0/shard_$source.pt" \
      "$SCRATCH/$name/s1/shard_$source.pt" --device cpu || return 1
    mv "$SCRATCH/$name/merge_$source/atlas.pt" "$SCRATCH/$name/atlas_$source.pt" || return
    cp "$SCRATCH/$name/merge_$source/merge.json" "$OUT/$name/merge_$source.json" || return
  done
}
copy_atlases() {
  "$PYTHON" - <<'PYATLASES'
import json
import os
import shutil
from pathlib import Path
import torch
from atlas.core import result_digests
root = Path(os.environ['OUT']) / 'atlases'
root.mkdir()
scratch = Path(os.environ['SCRATCH'])
digests = {}
for name in ('trained', 'shuf1', 'shuf2', 'shuf3'):
    run = json.loads((scratch / name / 's0/run.json').read_text())
    digests[name] = {}
    for source in ('neurons_post', 'randdir_512_0'):
        path = scratch / name / f'atlas_{source}.pt'
        atlas = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        digests[name][source] = {'digests': result_digests(atlas),
                                 'directions_sha256': run['sources'][source].get('directions_sha256')}
        del atlas
        if name == 'trained':
            shutil.copyfile(path, root / path.name)
(root / 'digests.json').write_text(json.dumps(digests, indent=2) + '\n')
PYATLASES
}
export -f record step preflight gpu_record smoke_descriptors inputs archive run_atlases copy_atlases

if [ "$SMOKE" = 1 ]; then
  ATLAS_THREADS=1
  MODEL=8M; DEV=cpu; DT=fp32; BACKEND=torch; TPB=4096; PT=8192; K=16; KR=16
  DATA=${SMOKE_DATA:-data/swissprot_test500.pt}
  HELD=lists/heldout_created_after_2021-11-17.tsv
  DESC="$OUT/smoke_inputs/descriptors.pt"
  UNIREF="$OUT/smoke_inputs/uniref50.tsv.gz"
  SMALL_COMPARE=${SMALL_COMPARE:-$OUT/compare.json}
  LIMITS=(--max-seqs 32); EDGES=(0.3 0.5 0.7 0.9 0.95 0.99)
  export DATA
  step smoke_descriptors smoke_descriptors || exit 1
  for name in preflight gpu_record smi_logger gpu_tests stats_gpu_tests; do skip "$name" 'SMOKE: CPU only'; done
else
  # As jobs/shard_scale/run.sh: weights cached on local scratch (never written to FS); one atlas process per GPU gets nproc / 8 threads.
  export HF_HOME=$SCRATCH/hf HF_HUB_CACHE=$SCRATCH/hf/hub
  ATLAS_THREADS=$(( $(nproc) / 8 > 0 ? $(nproc) / 8 : 1 ))
  MODEL=650M; DEV=cuda; DT=bf16; BACKEND=triton; TPB=32768; PT=262144; K=64; KR=256
  DATA=$FS/cache/swissprot_2026_03.pt
  HELD=heldout_created_after_2021-11-17.tsv
  DESC=$FS/descriptors_swissprot_2026_03.pt
  UNIREF=lists/swissprot_2026_03_uniref50.tsv.gz
  SMALL_COMPARE=${SMALL_COMPARE:-$JOB/results_8m_compare.json}
  LIMITS=(); EDGES=(0.5 0.9 0.99 0.999 0.9999 0.99999)
  step preflight preflight || exit 2
fi
export DATA DESC UNIREF HELD SMALL_COMPARE
step inputs inputs || exit 1
# Populate the shared HF cache once, before any timed GPU step.
DOWNLOAD=(env HF_HUB_OFFLINE=0)
if [ "$SMOKE" = 1 ]; then DOWNLOAD=(); fi
step model_download "${DOWNLOAD[@]}" "$PYTHON" -c "from atlas.esm import load; load('$MODEL', device='cpu'); print('model cached')" || exit 2
if [ "$SMOKE" != 1 ]; then
  step gpu_record gpu_record || exit 3
  Q=timestamp,index,uuid,power.draw,power.limit,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used
  echo "$Q" > "$OUT/smi.csv"
  nvidia-smi --query-gpu="$Q" --format=csv,noheader,nounits -l 1 >> "$OUT/smi.csv" 2> "$OUT/smi.err" &
  SMI_PID=$!
  ( while kill -0 "$SMI_PID" 2>/dev/null; do ps -o pid,ppid,etime,stat,cmd -p "$SMI_PID"; sleep 30; done ) > "$OUT/ps.log" 2>&1 &
  PS_PID=$!
  step smi_logger kill -0 "$SMI_PID" || exit 3
  step gpu_tests "$PYTHON" -m tests.test_triton_kernel --cuda || exit 5
  step stats_gpu_tests "$PYTHON" -m tests.test_stats_triton --cuda || exit 5
fi
COMMON=(--model "$MODEL" --data "$DATA" --device "$DEV" --dtype "$DT" --backend "$BACKEND"
        --tokens-per-batch "$TPB" --pilot-tokens "$PT" --K "$K" --Kr "$KR" --edge-q "${EDGES[@]}"
        --seed-thresholds --seed 0 --exclude-acc "$HELD" --prefetch 2 --stats-impl v3 --merge-impl stacked
        --no-gram --ckpt-every 1000000 --columns neurons_post,randdir:512:0 --source-cap neurons_post=0.04
        --deadline "$DEADLINE" "${LIMITS[@]}")
RUNS=(trained shuf1 shuf2 shuf3)
for i in "${!RUNS[@]}"; do
  run_atlases "${RUNS[$i]}" "$((2 * i))" "${COMMON[@]}" &
  PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
PIDS=()
[ "$FAILED" = 0 ] || exit 1
NAMES=(N R S1 S2 S3 RS1 RS2 RS3)
PATHS=("$SCRATCH/trained/atlas_neurons_post.pt" "$SCRATCH/trained/atlas_randdir_512_0.pt")
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
step summary "$PYTHON" scripts/compare_sources.py summarise --scores "$OUT/scores" \
  --pool N,R,S1,S2,S3,RS1,RS2,RS3 --out "$OUT/summary.json" || exit 1
step report "$PYTHON" scripts/compare_report.py --scores "$OUT/scores" --summary "$OUT/summary.json" \
  --out "$OUT/compare.json" || exit 1
step h3 "$PYTHON" scripts/h3_report.py --small "$SMALL_COMPARE" --large "$OUT/compare.json" \
  --out "$OUT/h3.json" || exit 1
# Include the generated stand-in's sha256 once it exists.
if [ "$SMOKE" = 1 ] && [ "$SMALL_COMPARE" = "$OUT/compare.json" ]; then
  "$PYTHON" - <<'PYINPUTS'
import json
import os
from pathlib import Path
from scripts.compare_sources import sha256
out = Path(os.environ['OUT'])
p = out / 'compare.json'
manifest = out / 'inputs.json'
rows = json.loads(manifest.read_text())
rows.append({'path': str(p), 'size': p.stat().st_size, 'sha256': sha256(p)})
manifest.write_text(json.dumps(rows, indent=2) + '\n')
PYINPUTS
  [ "$?" = 0 ] || exit 1
fi
step copy_atlases copy_atlases || exit 1
if [ "$SMOKE" != 1 ]; then run_step smi_alive_end kill -0 "$SMI_PID"; fi
exit "$FAILED"
