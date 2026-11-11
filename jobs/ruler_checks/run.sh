#!/bin/bash
# Frozen ruler checks. Env: JOB OUT FS DEADLINE; SMOKE=1 builds local inputs; POOLING=0 skips pooling.
set -uo pipefail
: "${JOB:?}" "${OUT:?}" "${FS:?}" "${DEADLINE:?}"
PYTHON=${PYTHON:-python3}
SMOKE=${SMOKE:-0}
POOLING=${POOLING:-1}
WORKERS=${WORKERS:-$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))}
SHAPES_IN=${SHAPES:-}  # SHAPES=none: pooling check only (no planted shapes)
SHAPES=${SHAPES:-"gauss gelu gelu_off randdir sae:0.1 sae:0.01 sae:0.001"}
PER_LAYER=${PER_LAYER:-250}
NOISE=500
if [ "$SMOKE" = 1 ]; then
  [ "$SHAPES_IN" = none ] || SHAPES="gauss sae:0.01"; PER_LAYER=8; NOISE=20
  [ "$WORKERS" -le 4 ] || WORKERS=4
fi
[[ "$POOLING" =~ ^[01]$ ]] || exit 2
[[ "$DEADLINE" =~ ^[0-9]+$ && "$WORKERS" =~ ^[1-9][0-9]*$ && "$PER_LAYER" =~ ^[1-9][0-9]*$ ]] || exit 2
OUT=$("$PYTHON" - "$OUT" "$SHAPES" <<'PYOUT'
import sys
from pathlib import Path
p = Path(sys.argv[1]).absolute()
shapes = sys.argv[2].split()
if shapes == ['none']:
    shapes = []
elif not shapes or len(shapes) != len(set(shapes)):
    sys.exit("SHAPES must be nonempty and distinct")
for shape in shapes:
    if shape in ('gauss', 'gelu', 'gelu_off', 'randdir', 'sae'):
        continue
    if not shape.startswith('sae:') or not shape[4:].replace('.', '', 1).isdigit() or not 0 < float(shape[4:]) < 1:
        sys.exit(f"invalid shape: {shape}")
RUNNER_FILES = {'started.json', 'job.log'}  # written into OUT by infra/lambda_job.py's inbox runner before run.sh starts; empty directories are also accepted
if p.is_symlink() or (p.exists() and (not p.is_dir() or any(q.name not in RUNNER_FILES and not (q.is_dir() and not any(q.iterdir())) for q in p.iterdir()))):
    sys.exit("OUT must be a new or empty directory, not a symlink")
print(p.resolve())
PYOUT
) || exit 2
mkdir -p "$OUT/timing" || exit 2
export OUT JOB PYTHON SMOKE DEADLINE WORKERS PER_LAYER NOISE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR="$OUT/tmp" TRITON_CACHE_DIR="$OUT/tmp/triton" TORCHINDUCTOR_CACHE_DIR="$OUT/tmp/inductor"
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

smoke_descriptors() {
  "$PYTHON" - <<'PY'
import gzip
import os
from pathlib import Path
import torch
root = Path(os.environ['OUT']) / 'smoke_inputs'
(root / 'desc_subset').mkdir(parents=True)
data = torch.load('data/swissprot_test500.pt', map_location='cpu', weights_only=False)
# Retain original sequence indices in packed token ids.
indices = sorted(range(len(data['seq'])), key=lambda i: (len(data['seq'][i]), i))[:32]
acc = [data['acc'][i] for i in indices]
seq = {i: data['seq'][i] for i in indices}
torch.save({'seq': seq, 'acc': dict(zip(indices, acc))}, root / 'swissprot.pt')
(root / 'accessions.txt').write_text('\n'.join(acc) + '\n')
lengths = [len(seq[i]) for i in indices]
offset = torch.tensor([0] + lengths).cumsum(0)
n = int(offset[-1])
g = torch.Generator().manual_seed(0)
d = {'acc': acc, 'seq_idx': torch.tensor(indices), 'offset': offset,
     'heldout': torch.zeros(len(indices), dtype=torch.bool),
     'ss3': torch.randint(3, (n,), generator=g),
     'plddt': torch.full((n,), 90.), 'burial': torch.randint(46, (n,), generator=g).float(),
     'phi': torch.rand(n, generator=g) * 360 - 180, 'psi': torch.rand(n, generator=g) * 360 - 180}
torch.save(d, root / 'desc_subset' / 'part_000.pt')
torch.save({k: v for k, v in d.items() if k != 'heldout'}, root / 'descriptors.pt')
with gzip.open(root / 'uniref50.tsv.gz', 'wt') as f:
    f.write('Entry\tUniRef50\n')
    for a in acc:
        f.write(f'{a}\tUniRef50_{a}\n')
PY
}

smoke_atlas() {
  "$PYTHON" - "$1" <<'PY'
import os
import sys
from pathlib import Path
import torch
from atlas.core import Atlas, AtlasConfig, pack_tok
from atlas.reference import pilot_edges
root = Path(os.environ['OUT']) / 'smoke_inputs'
source = sys.argv[1]
seed = ('trained', 'shuf1', 'shuf2', 'shuf3').index(source)
data = torch.load(root / 'swissprot.pt', map_location='cpu', weights_only=False)
lengths = [len(seq) for seq in data['seq'].values()]
tok = torch.cat([pack_tok(torch.full((len(seq),), i), torch.arange(len(seq)))
                 for i, seq in data['seq'].items()])
g = torch.Generator().manual_seed(seed)
Hs = [torch.randn(tok.numel(), 32, generator=g) for _ in range(2)]
# Wider bins and Kr >= 10 let the tiny fixture populate the ruler's peak scores.
edges = pilot_edges(Hs, probs=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95))
at = Atlas(2, 32, edges, AtlasConfig(K=16, Kr=16, backend='torch', seed=seed, gram=False), n_annot=0)
n_batches = 0
for start in range(0, tok.numel(), 4096):
    tk = tok[start:start + 4096]
    ctx = at.make_ctx(tk.numel(), torch.arange(tk.numel()), tk, None)
    at.begin_batch(ctx)
    for l, H in enumerate(Hs):
        at.observe(l, H[start:start + 4096])
    redo = at.end_batch()
    if redo:
        at.begin_batch(ctx, redo_layers=redo)
        for l in redo:
            at.observe(l, Hs[l][start:start + 4096])
        assert at.end_batch() == []
    n_batches += 1
res = at.result()
res.update(model={'name': 'smoke', 'n_layers': 2, 'd_ff': 32},
           coverage={'n_seq': len(lengths), 'n_res_full': sum(lengths), 'n_res_used': sum(lengths),
                     'n_truncated': 0, 'frac_seq_truncated': 0., 'frac_res_dropped': 0.},
           vocab=[], order_seed=0, n_total_plan=tok.numel(), n_batches=n_batches)
(root / source).mkdir()
torch.save(res, root / source / 'atlas.pt')
PY
}

inputs() {
  "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path
from scripts.compare_sources import sha256
paths = {name: Path(os.environ[name]) for name in ('DATA', 'DESC_SUBSET', 'DESC_WHOLE', 'UNIREF',
                                                  'A8_PRE', 'A8_SHUF1', 'A8_SHUF2', 'A8_SHUF3')}
result = {}
for name, p in paths.items():
    if not p.exists():
        raise FileNotFoundError(p)
    files = sorted(p.glob('part_*.pt')) if p.is_dir() else [p]
    result[name] = {'path': str(p), 'size': sum(f.stat().st_size for f in files), 'sha256': sha256(p)}
(Path(os.environ['OUT']) / 'inputs.json').write_text(json.dumps(result, indent=2))
PY
}

build_columns() {
  "$PYTHON" - "$@" <<'PY'
import runpy
import sys
import torch
torch.set_num_threads(int(sys.argv.pop(1)))
sys.argv[0] = 'scripts/synthetic_columns.py'
runpy.run_path(sys.argv[0], run_name='__main__')
PY
}

planted() {
  local shape=$1 threads=$2 dir="$OUT/planted/$1" base=${1%%:*} count
  local -a extra=()
  [[ "$shape" != sae:* ]] || extra=(--sae-rate "${shape#*:}")
  mkdir -p "$dir"
  step "planted_${shape}_build" build_columns "$threads" --data "$DATA" --desc-dir "$DESC_SUBSET" \
    --gen-workers "$threads" --uniref "$UNIREF" --shape "$base" "${extra[@]}" --mixtures --noise-columns "$NOISE" --out "$dir" || return
  count=$("$PYTHON" - "$dir/columns.json" <<'PY'
import json
import sys
print(len(json.load(open(sys.argv[1]))))
PY
) || return
  step "planted_${shape}_sample" "$PYTHON" scripts/compare_sources.py sample --atlas "$shape=$dir/synthetic_atlas.pt" \
    --per-layer "$count" --seed 0 --out "$dir/sample.json" || return
  step "planted_${shape}_score" "$PYTHON" scripts/compare_sources.py score --sample "$dir/sample.json" --source "$shape" \
    --atlas "$dir/synthetic_atlas.pt" --desc "$DESC_SUBSET" --data "$DATA" --uniref "$UNIREF" \
    --workers "$threads" --out "$dir/scores" || return
  step "planted_${shape}_calls" "$PYTHON" scripts/compare_sources.py calls --scores "$dir/scores" --source "$shape" \
    --rule v1.2 --out "$dir/calls.json" || return
  step "planted_${shape}_eval" "$PYTHON" scripts/eval_synthetic.py --calls "$dir/calls.json" \
    --columns "$dir/columns.json" --out "$dir/gate.json" || return
  step "planted_${shape}_summary" "$PYTHON" scripts/compare_sources.py summarise --scores "$dir/scores" \
    --pool "$shape" --out "$dir/summary.json"
}

smoke_parity() {
  local dir="$OUT/planted/gauss"
  "$PYTHON" scripts/run_ruler.py --atlas "$dir/synthetic_atlas.pt" --data "$DATA" --desc-dir "$DESC_SUBSET" \
    --uniref "$UNIREF" --rule v1.2 --out "$dir/reference_calls.json" || return
  "$PYTHON" - "$dir" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
a = {c['col']: c for c in json.loads((p / 'calls.json').read_text())['calls']}
b = {c['col']: c for c in json.loads((p / 'reference_calls.json').read_text())['calls']}
assert a == b, 'adapter and run_ruler.py differ'  # adapter keeps sample order; compare record by record via col
print('adapter equals run_ruler.py --rule v1.2')
PY
}

export -f timeline record skip step run_step smoke_descriptors smoke_atlas inputs build_columns planted smoke_parity
if [ "$SMOKE" = 1 ]; then
  step smoke_descriptors smoke_descriptors || exit 1
  DATA="$OUT/smoke_inputs/swissprot.pt"
  DESC_SUBSET="$OUT/smoke_inputs/desc_subset"
  DESC_WHOLE="$OUT/smoke_inputs/descriptors.pt"
  UNIREF="$OUT/smoke_inputs/uniref50.tsv.gz"
  for source in trained shuf1 shuf2 shuf3; do
    step "smoke_atlas_$source" smoke_atlas "$source" || exit 1
  done
  A8_PRE="$OUT/smoke_inputs/trained/atlas.pt"
  A8_SHUF1="$OUT/smoke_inputs/shuf1/atlas.pt"
  A8_SHUF2="$OUT/smoke_inputs/shuf2/atlas.pt"
  A8_SHUF3="$OUT/smoke_inputs/shuf3/atlas.pt"
else
  DATA="$FS/cache/swissprot_2026_03.pt"
  DESC_SUBSET="$FS/runs/atlas-wab-20261105-0140/desc_subset"
  DESC_WHOLE="$FS/descriptors_swissprot_2026_03.pt"
  UNIREF="lists/swissprot_2026_03_uniref50.tsv.gz"
  A8_PRE="$FS/runs/atlas-batch-20261105-0258/a8_pre/atlas.pt"
  A8_SHUF1="$FS/runs/atlas-batch2-20261105-0348/a8_shuf1/atlas.pt"
  A8_SHUF2="$FS/runs/atlas-batch-20261105-0258/a8_shuf2/atlas.pt"
  A8_SHUF3="$FS/runs/atlas-batch2-20261105-0348/a8_shuf3/atlas.pt"
fi
export DATA DESC_SUBSET DESC_WHOLE UNIREF A8_PRE A8_SHUF1 A8_SHUF2 A8_SHUF3
step inputs inputs || exit 1
mkdir -p "$OUT/planted"
shapes=()
[ "$SHAPES" = none ] || read -ra shapes <<< "$SHAPES"
[ "${#shapes[@]}" -gt 0 ] || [ "$SHAPES" = none ] || exit 2
threads=$(( WORKERS / (${#shapes[@]} > 0 ? ${#shapes[@]} : 1) )); [ "$threads" -gt 0 ] || threads=1
for shape in "${shapes[@]}"; do
  planted "$shape" "$threads" &
  PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
PIDS=()
[ "$FAILED" = 0 ] || exit 1
if [ "$SMOKE" = 1 ] && [ "$SHAPES" != none ]; then step smoke_parity smoke_parity || exit 1; fi

if [ "$POOLING" = 1 ]; then
  mkdir -p "$OUT/pooling"
  step pooling_sample "$PYTHON" scripts/compare_sources.py sample --atlas "trained=$A8_PRE" "shuf1=$A8_SHUF1" \
    "shuf2=$A8_SHUF2" "shuf3=$A8_SHUF3" --per-layer "$PER_LAYER" --seed 0 --out "$OUT/pooling/sample.json" || exit 1
  for source in trained shuf1 shuf2 shuf3; do
    case "$source" in
      trained) atlas=$A8_PRE;; shuf1) atlas=$A8_SHUF1;; shuf2) atlas=$A8_SHUF2;; shuf3) atlas=$A8_SHUF3;;
    esac
    step "pooling_score_$source" "$PYTHON" scripts/compare_sources.py score --sample "$OUT/pooling/sample.json" \
      --source "$source" --atlas "$atlas" --desc "$DESC_WHOLE" --data "$DATA" --uniref "$UNIREF" \
      --workers "$WORKERS" --out "$OUT/pooling/scores" || exit 1
  done
  step pooling_score_trained_4 "$PYTHON" scripts/compare_sources.py score --sample "$OUT/pooling/sample.json" \
    --source trained --atlas "$A8_PRE" --desc "$DESC_WHOLE" --data "$DATA" --uniref "$UNIREF" \
    --workers 4 --out "$OUT/pooling/scores_4" || exit 1
  for pool in trained trained,shuf1 trained,shuf1,shuf2,shuf3; do
    step "pooling_summary_$pool" "$PYTHON" scripts/compare_sources.py summarise --scores "$OUT/pooling/scores" \
      --pool "$pool" --out "$OUT/pooling/summary_$pool.json" || exit 1
  done
fi
step report "$PYTHON" scripts/ruler_checks_report.py "$OUT" || exit 1
