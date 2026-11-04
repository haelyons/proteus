#!/bin/bash
# ESM-2 650M atlas over Swiss-Prot 2026_03 minus the post-ESM-2 held-out set, one GPU.
# Env: JOB OUT FS DEADLINE. SMOKE=1 runs the same steps with ESM-2 8M on CPU (policy: nothing launches
# that has not run end to end somewhere); in the smoke, run B uses the Triton kernels in the interpreter.
#   1. GPU tests: compaction kernel, stats/Gram kernels at the 650M shape (with timings), CUDA exactness
#   2. equivalence gate on real data, frozen config: A = reference (stats v1, merge loop), B = new
#      (stats v3, merge stacked), same seeds and batches, checkpoints at GATE batches compared
#      (exact fields bitwise incl. top-K ids, sums rel diff < 1e-5). The full pass runs only if it holds.
#   3. full pass: frozen config, fresh pilot, overhead on every PAIR-th batch, checkpoints, profiles
# Frozen config: K=64, Kr=64, bin edges at pilot q (0.5 0.9 0.99 0.999 0.9999 0.99999), seeded
# thresholds, rho_sub 0.05, rho_res 0.01, held-out (UniProt entries created after 2021-11-17) excluded.
set -uo pipefail
SMOKE=${SMOKE:-0}
HELDOUT=heldout_created_after_2021-11-17.tsv
EDGES="0.5 0.9 0.99 0.999 0.9999 0.99999"
if [ "$SMOKE" = 1 ]; then
  MODEL=8M; DEV=cpu; DT=fp32; BACKEND=torch; TPB=4096; PT=65536; GATE=6; PAIR=5; PROF="10"; CKPT=10; FLUSH=10; MAXB="--max-batches 20"
  export OMP_NUM_THREADS=1  # CPU SDPA is not bitwise reproducible multi-threaded; the gate compares bitwise
  DATA=${SMOKE_DATA:-data/swissprot_2026_03.pt}; HELD=${SMOKE_HELD:-lists/$HELDOUT}
else
  MODEL=650M; DEV=cuda; DT=bf16; BACKEND=triton; TPB=32768; PT=262144; GATE=60; PAIR=25; PROF="500 3000"; CKPT=1500; FLUSH=1000; MAXB=""
  DATA=$FS/cache/swissprot_2026_03.pt; HELD=$HELDOUT
fi
echo "[atlas] start $(date -u +%FT%TZ) job=$JOB smoke=$SMOKE deadline=$DEADLINE ($(( DEADLINE - $(date +%s) ))s left)"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SMI_PID=""
if [ "$SMOKE" != 1 ]; then
  export HF_HOME=$FS/cache/hf
  nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit,clocks.max.sm --format=csv,noheader
  free -g | head -2
  python3 -m pip install --user -q -U "pillow>=10" "transformers>=4.46" 2>&1 | tail -1
  python3 -c "import torch, triton, transformers, PIL; from transformers import EsmModel; print('[atlas] versions torch', torch.__version__, 'triton', triton.__version__, 'transformers', transformers.__version__, '| EsmModel import ok')" || { echo "[atlas] preflight FAILED"; exit 2; }
  Q="timestamp,power.draw,power.limit,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used"
  echo "$Q" > $OUT/smi.csv
  nvidia-smi --query-gpu=$Q --format=csv,noheader,nounits -l 1 >> $OUT/smi.csv 2>&1 &
  SMI_PID=$!
  echo "[atlas] gpu tests"
  python3 -m tests.test_triton_kernel --cuda > $OUT/gpu_tests.log 2>&1; r1=$?
  python3 -m tests.test_stats_triton --cuda > $OUT/stats_kernel_tests.log 2>&1; r2=$?
  echo "[atlas] gpu tests rc=$r1/$r2 | $(grep -E '\] (OK|FAILED)' $OUT/gpu_tests.log | cut -c1-90 | tr '\n' ' ') | $(grep -E 'OK|FAILED' $OUT/stats_kernel_tests.log | tr '\n' ' ' | cut -c1-900)"
  if [ $r1 != 0 ] || [ $r2 != 0 ]; then echo "[atlas] GPU tests failed: stopping before the gate"; kill $SMI_PID 2>/dev/null; exit 5; fi
fi
if [ ! -f $DATA ] || [ ! -f $HELD ]; then echo "[atlas] missing $DATA or $HELD"; exit 3; fi

FROZEN="--model $MODEL --data $DATA --device $DEV --dtype $DT --backend $BACKEND --tokens-per-batch $TPB --pilot-tokens $PT
        --K 64 --Kr 64 --edge-q $EDGES --seed-thresholds --exclude-acc $HELD --deadline $DEADLINE"
echo "[atlas] gate A: stats v1, merge loop ($GATE batches)"
python3 scripts/run_atlas.py $FROZEN --out $OUT/gate_A --stats-impl v1 --merge-impl loop --stop-after $GATE --ckpt-every $GATE --pair-every 1 \
  || echo "[atlas] gate A rc=$?"
echo "[atlas] gate B: stats v3, merge stacked ($GATE batches)"
if [ "$SMOKE" = 1 ]; then export TRITON_INTERPRET=1; fi
python3 scripts/run_atlas.py $FROZEN --out $OUT/gate_B --stats-impl v3 --merge-impl stacked --stop-after $GATE --ckpt-every $GATE --pair-every 1 \
  --profile-at $(( GATE - 5 )) || echo "[atlas] gate B rc=$?"
unset TRITON_INTERPRET
python3 scripts/compare_atlas.py $OUT/gate_A/ckpt/ckpt_$GATE.pt $OUT/gate_B/ckpt/ckpt_$GATE.pt --out $OUT/gate_compare.json > $OUT/gate_compare.log 2>&1; gate=$?
echo "[atlas] gate rc=$gate $(python3 -c "import json; r=json.load(open('$OUT/gate_compare.json')); print('ok', r['ok'], {k: v for k, v in r['fields'].items() if not (v.get('exact') or v.get('equal'))})" 2>&1 | cut -c1-700)"
python3 - <<EOF 2>&1 | sed 's/^/[atlas] /'
import json, statistics as st
for X in ("gate_A", "gate_B"):
    R = [json.loads(l) for l in open("$OUT/" + X + "/batches.jsonl")]
    o = [r["overhead_vs_fwd"] for r in R if r.get("overhead_vs_fwd") is not None and not r["redo_layers"] and not r["profiled"]]
    if o: print(f"{X}: {len(R)} batches, overhead (no redo) mean {st.mean(o):.3f} median {st.median(o):.3f}; fwd {st.mean(r['fwd_only_s'] for r in R):.3f}s; redo {[r['batch'] for r in R if r['redo_layers']]}")
EOF
if [ $gate != 0 ]; then echo "[atlas] equivalence gate FAILED: full pass not started"; [ -n "$SMI_PID" ] && kill $SMI_PID 2>/dev/null; exit 6; fi

echo "[atlas] full pass (stats v3, merge stacked)"
t0=$(date +%s)
python3 scripts/run_atlas.py $FROZEN --out $OUT/full $MAXB --pair-every $PAIR --ckpt-every $CKPT --gram-flush-every $FLUSH --profile-at $PROF; rc=$?
echo "[atlas] full pass rc=$rc wall $(( $(date +%s) - t0 ))s"
python3 - <<EOF 2>&1 | sed 's/^/[atlas] /'
import json, statistics as st
R = [json.loads(l) for l in open("$OUT/full/batches.jsonl")]
j = json.load(open("$OUT/full/run.json"))
def mean(xs): return st.mean(xs) if xs else float("nan")
def median(xs): return st.median(xs) if xs else float("nan")
p = [r for r in R if r.get("fwd_only_s") is not None]
o = [r["overhead_vs_fwd"] for r in p if not (r["redo_layers"] or r["profiled"] or r.get("gram_flushed") or r.get("ckpt_s"))]
fwd = mean([r["fwd_only_s"] for r in p])
atl = sum(r["atlas_pass_s"] for r in R)
print(f"batches {len(R)} of {j['n_batches_plan']}, complete {j['complete']}; tokens {j['n_seen']:,}; excluded seqs {j.get('n_excluded')}")
print(f"seconds: load {j['seconds']['load']:.0f}, pilot {j['seconds']['pilot']:.0f}, stream {j['seconds']['stream_this_process']:.0f}; final save {j.get('final_save_s')}")
print(f"atlas passes {atl:.0f}s; paired batches {len(p)}: hook-free forward mean {fwd:.3f}s; overhead mean {mean(o):.3f} median {median(o):.3f} (n={len(o)}, excl. redo/profiled/flush/ckpt)")
print(f"estimated forward-only pass {fwd * len(R):.0f}s; redo batches {sum(1 for r in R if r['redo_layers'])}; checkpoints {[round(r['ckpt_s']) for r in R if r.get('ckpt_s')]}")
print(f"seed verification: {j.get('res_seed_limited_bins')} of {j.get('res_seeded_bins')} seeded bins short; top-K groups short {j.get('top_seed_short_groups')}; gemm {json.dumps(j.get('gemm_paths'))[:200]}")
EOF
[ -n "$SMI_PID" ] && kill $SMI_PID 2>/dev/null
echo "[atlas] end $(date -u +%FT%TZ) rc=$rc"
exit $rc
