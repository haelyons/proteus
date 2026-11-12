#!/bin/bash
# Frozen CPU unit study. Env: JOB OUT FS DEADLINE; see specs/units.md.
set -uo pipefail
if [ "${SMOKE:-0}" = 1 ]; then
  JOB=${JOB:-units-smoke}
  OUT=${OUT:-$PWD/units_smoke}
  FS=${FS:-$PWD}
  DEADLINE=${DEADLINE:-$(( $(date +%s) + 1800 ))}
fi
: "${JOB:?}" "${OUT:?}" "${FS:?}" "${DEADLINE:?}"
PYTHON=${PYTHON:-python3}
SMOKE=${SMOKE:-0}
CALIB=${CALIB:-$FS/runs/calib-8m-20261112-1018}
SCRATCH=${SCRATCH:-$PWD/.units_scratch_$JOB}
WORKERS=${WORKERS:-$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))}
if [ "$SMOKE" = 1 ] && [ "$WORKERS" -gt 2 ]; then WORKERS=2; fi
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$DEADLINE" =~ ^[0-9]+$ ]] || { echo "invalid DEADLINE"; exit 2; }
# Require fresh OUT and reject symlinks that could redirect writes.
export SMOKE
PATHS=$("$PYTHON" - "$OUT" "$SCRATCH" "$FS" <<'PY'
import sys
import os
from pathlib import Path
out, scratch, fs = [Path(p).absolute() for p in sys.argv[1:]]
RUNNER_FILES = {'started.json', 'job.log'}  # written into OUT by infra/lambda_job.py's inbox runner before run.sh starts
if out.is_symlink() or (out.exists() and (not out.is_dir() or any(p.name not in RUNNER_FILES for p in out.iterdir()))):
    sys.exit("OUT must be a new or empty directory, not a symlink")
if scratch.exists() or scratch.is_symlink():
    sys.exit("SCRATCH must be a new directory, not a symlink")
for p in (out, scratch):
    if not p.parent.is_dir():
        sys.exit("OUT and SCRATCH parents must exist")
out, scratch, fs = out.resolve(), scratch.resolve(), fs.resolve()
if out == scratch or out in scratch.parents or scratch in out.parents:
    sys.exit("OUT and SCRATCH must be separate directories")
if not os.environ.get("SMOKE") == "1" and (scratch == fs or fs in scratch.parents):
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
export PYTHONUSERBASE="$SCRATCH/python"
mkdir -p "$TMPDIR"
PIDS=(); FAILED=0

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
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  timeline
  rm -rf -- "$SCRATCH"
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
  "$PYTHON" -m pip install --user -q -U 'pillow>=10' 'transformers>=4.46' pytest || return
  "$PYTHON" -c 'import torch, triton, transformers, PIL, pytest; from transformers import EsmModel; print("preflight ok")'
}

smoke_inputs() {
  "$PYTHON" - <<'PY'
import gzip
import os
from pathlib import Path
import torch
root = Path(os.environ['OUT']) / 'smoke_inputs'
root.mkdir()
(root / 'desc_subset').mkdir()
data = torch.load(os.environ['SMOKE_DATA'], map_location='cpu', weights_only=False)
# Short, disjoint stand-ins: 64 discovery proteins and 40 held-out proteins.
data['seq'] = [s[:64] for s in data['seq'][:104]]
data['acc'] = data['acc'][:104]
data['len'] = torch.tensor([len(s) for s in data['seq']])
data['iv'] = data['iv'][data['iv'][:, 0] < 104]
torch.save(data, root / 'data.pt')
offset = torch.tensor([0] + data['len'].tolist()).cumsum(0)
n = int(offset[-1])
g = torch.Generator().manual_seed(0)
d = {'acc': data['acc'], 'seq_idx': torch.arange(104), 'offset': offset,
     'heldout': torch.arange(104) >= 64, 'ss3': torch.randint(4, (n,), generator=g) - 1,
     'plddt': torch.where(torch.rand(n, generator=g) < .2, 60., 90.),
     'burial': torch.randint(46, (n,), generator=g).float(),
     'phi': torch.rand(n, generator=g) * 360 - 180,
     'psi': torch.rand(n, generator=g) * 360 - 180}
torch.save(d, root / 'descriptors.pt')
torch.save(d, root / 'desc_subset/part_000.pt')
with gzip.open(root / 'uniref50.tsv.gz', 'wt') as f:
    f.write('Entry\tUniRef50\n')
    for acc in data['acc']:
        f.write(f'{acc}\tUniRef50_{acc}\n')
(root / 'heldout.tsv').write_text('\n'.join(data['acc'][64:]) + '\n')
PY
}

inputs() {
  "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path
from scripts.compare_sources import sha256
paths = [Path(os.environ[k]) for k in ('DATA', 'DESC_DIR', 'DESC', 'UNIREF')]
paths += [Path(os.environ['CALIB']) / 'atlases' / name for name in
          ('atlas_neurons_post.pt', 'atlas_randdir_512_0.pt')]
# Calibration archives the recorded direction digest and quantile labels here.
meta = [Path(os.environ['CALIB']) / p for p in ('atlases/run.json', 'trained/run.json')]
paths.append(next((p for p in meta if p.exists()), meta[-1]))
paths += [Path(os.environ['CASES'])]
rows = []
for p in paths:
    if not p.exists():
        raise FileNotFoundError(p)
    rows.append({'path': str(p), 'size': p.stat().st_size if p.is_file() else
                 sum(f.stat().st_size for f in p.glob('part_*.pt')), 'sha256': sha256(p)})
(Path(os.environ['OUT']) / 'inputs.json').write_text(json.dumps(rows, indent=2) + '\n')
PY
}
export -f timeline record skip step preflight smoke_inputs inputs

if [ "$SMOKE" = 1 ]; then
  export SMOKE_DATA=${SMOKE_DATA:-data/swissprot_test500.pt}
  step smoke_inputs smoke_inputs || exit 1
  DATA="$OUT/smoke_inputs/data.pt"
  DESC_DIR="$OUT/smoke_inputs/desc_subset"
  DESC="$OUT/smoke_inputs/descriptors.pt"
  UNIREF="$OUT/smoke_inputs/uniref50.tsv.gz"
  CALIB="$OUT/smoke_inputs"
  step smoke_atlas "$PYTHON" scripts/run_atlas.py --model 8M --data "$DATA" \
    --device cpu --dtype fp32 --backend torch --tokens-per-batch 4096 --K 16 --Kr 8 \
    --max-seqs 64 --pilot-tokens 4096 --edge-q .5 .9 .99 .999 .9999 .99999 \
    --exclude-acc "$OUT/smoke_inputs/heldout.tsv" --columns neurons_post,randdir:512:0 \
    --no-gram --deadline "$DEADLINE" --out "$CALIB/atlases" || exit 1
  # Independent legacy implementation supplies smoke reproduction references.
  step smoke_unit_writes "$PYTHON" scripts/unit_writes.py --data "$DATA" --proteins 2 \
    --units 1 --max-res 64 --out "$OUT/smoke_inputs/unit_writes" || exit 1
  CASES="$OUT/smoke_inputs/unit_writes/cases.jsonl"
  FLAGS=(--smoke)
else
  export HF_HOME=$FS/cache/hf
  DATA=$FS/cache/swissprot_2026_03.pt
  DESC_DIR=$FS/runs/atlas-wab-20261105-0140/desc_subset
  DESC=$FS/descriptors_swissprot_2026_03.pt
  UNIREF=lists/swissprot_2026_03_uniref50.tsv.gz
  CASES=${CASES:-${UNIT_WRITES_CASES:-results/unit_writes_8M_20261105/cases.jsonl}}
  # The launcher packs included directories at the job root under their basename.
  if [ ! -e results/unit_writes_8M_20261105 ] && [ -f unit_writes_8M_20261105/cases.jsonl ]; then
    mkdir -p results && ln -s ../unit_writes_8M_20261105 results/unit_writes_8M_20261105 || exit 1
  fi
  step preflight preflight || exit 2
  FLAGS=()
fi
export CALIB DATA DESC_DIR DESC UNIREF CASES
step inputs inputs || exit 1
COMMON=(--out "$OUT" --scratch "$SCRATCH" --calib "$CALIB" --data "$DATA" \
        --desc-dir "$DESC_DIR" --desc "$DESC" --uniref "$UNIREF" --cases "$CASES" \
        --workers "$WORKERS" "${FLAGS[@]}")
for name in repro_unit_writes sample heldout_values columns report; do
  step "$name" "$PYTHON" scripts/units_study.py "$name" "${COMMON[@]}" || exit 1
  if [ "$name" = repro_unit_writes ]; then
    if [ "$SMOKE" = 1 ]; then
      skip blind_repro_test 'SMOKE: requires full Swiss-Prot cache'
    else
      run_step blind_repro_test env PROTEUS_SWISSPROT="$DATA" "$PYTHON" -m pytest -q \
        -p no:cacheprovider tests/test_intervene.py -k reproduction
    fi
  fi
done
exit 0
