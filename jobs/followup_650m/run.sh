#!/bin/bash
# Follow-up to science_650m on one A100 40 GB. Env: JOB OUT FS DEADLINE. SMOKE=1: same steps, ESM-2 8M, CPU.
# Two engineering changes only, everything else as in the frozen config:
#   (a) batches and their host-side contexts built on a prefetch thread (--prefetch 2)
#   (b) stats-kernel row split (R chosen automatically: ~4 programs per SM)
# Steps: GPU tests; row-split sweep (stats kernel only); equivalence gate (v1 + loop vs v3 + stacked, 60
# batches); full pass (timing for the README; atlas copy #2 with per-field digests); held-out atlas (same
# bin edges); redundancy (distinct proteins / UniRef50 clusters per bin); AlphaFold DB descriptors for
# every accession of the build (held-out included). nvidia-smi logger checked with ps throughout (ps.log).
set -uo pipefail
SMOKE=${SMOKE:-0}
HELDOUT=heldout_created_after_2021-11-17.tsv
UNIREF=swissprot_2026_03_uniref50.tsv.gz
EDGES="0.5 0.9 0.99 0.999 0.9999 0.99999"
if [ "$SMOKE" = 1 ]; then
  MODEL=8M; DEV=cpu; DT=fp32; BACKEND=torch; TPB=4096; PT=65536; GATE=6; PAIR=5; PROF="10"; CKPT=10; FLUSH=10
  MAXB="--max-batches 20"; HMAXB="--max-batches 3"; NDESC="--n 20 --threads 8"
  export OMP_NUM_THREADS=1
  DATA=${SMOKE_DATA:-data/swissprot_2026_03.pt}; HELD=lists/$HELDOUT; UNI=${SMOKE_UNI:-lists/$UNIREF}; NPROC=2
else
  MODEL=650M; DEV=cuda; DT=bf16; BACKEND=triton; TPB=32768; PT=262144; GATE=60; PAIR=25; PROF="500 3000"; CKPT=1500; FLUSH=1000
  MAXB=""; HMAXB=""; NDESC=""
  DATA=$FS/cache/swissprot_2026_03.pt; HELD=$HELDOUT; UNI=$UNIREF; NPROC=$(( $(nproc) - 2 ))
fi
left() { echo $(( DEADLINE - $(date +%s) )); }
echo "[atlas] start $(date -u +%FT%TZ) job=$JOB smoke=$SMOKE deadline=$DEADLINE ($(left)s left)"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SMI_PID=""; PS_PID=""
if [ "$SMOKE" != 1 ]; then
  export HF_HOME=$FS/cache/hf
  nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit,clocks.max.sm --format=csv,noheader
  echo "[atlas] host: $(nproc) cpus, $(free -g | awk '/Mem/{print $2}') GB RAM, local disk free $(df -BG --output=avail $HOME | tail -1)"
  python3 -m pip install --user -q -U "pillow>=10" "transformers>=4.46" "pydssp" 2>&1 | tail -1
  python3 -c "import torch, triton, transformers, PIL, pydssp; from transformers import EsmModel; print('[atlas] versions torch', torch.__version__, 'triton', triton.__version__, 'transformers', transformers.__version__, '| ok')" || { echo "[atlas] preflight FAILED"; exit 2; }
  Q="timestamp,power.draw,power.limit,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used"
  echo "$Q" > $OUT/smi.csv
  nvidia-smi --query-gpu=$Q --format=csv,noheader,nounits -l 1 >> $OUT/smi.csv 2> $OUT/smi.err &
  SMI_PID=$!
  ( while true; do echo "== $(date -u +%T) smi.csv lines $(wc -l < $OUT/smi.csv)"; ps -o pid,ppid,etime,stat,cmd -p $SMI_PID 2>&1 | tail -n +1; ps -eo pid,etime,stat,cmd | grep -i [n]vidia-smi; sleep 120; done ) > $OUT/ps.log 2>&1 &
  PS_PID=$!
  sleep 3; echo "[atlas] ps nvidia-smi logger (pid $SMI_PID): $(ps -o pid,etime,stat,cmd -p $SMI_PID | tail -1)"
  echo "[atlas] gpu tests"
  python3 -m tests.test_triton_kernel --cuda > $OUT/gpu_tests.log 2>&1; r1=$?
  python3 -m tests.test_stats_triton --cuda > $OUT/stats_kernel_tests.log 2>&1; r2=$?
  echo "[atlas] gpu tests rc=$r1/$r2 | $(grep -E 'OK|FAILED' $OUT/stats_kernel_tests.log | grep -c OK) stats/gram cases OK"
  if [ $r1 != 0 ] || [ $r2 != 0 ]; then echo "[atlas] GPU tests failed"; kill $SMI_PID $PS_PID 2>/dev/null; exit 5; fi
  python3 scripts/tune_stats.py --rows-only --out $OUT/tune_rows.json 2>&1 | grep "by R" | sed 's/^/[atlas] /'
  echo "[atlas] ps nvidia-smi after tests: $(ps -o pid,etime,stat,cmd -p $SMI_PID | tail -1) | smi.csv lines $(wc -l < $OUT/smi.csv)"
fi
for f in $DATA $HELD $UNI; do [ -f $f ] || { echo "[atlas] missing $f"; exit 3; }; done

FROZEN="--model $MODEL --data $DATA --device $DEV --dtype $DT --backend $BACKEND --tokens-per-batch $TPB --pilot-tokens $PT
        --K 64 --Kr 64 --edge-q $EDGES --seed-thresholds --exclude-acc $HELD --deadline $DEADLINE --prefetch 2"
echo "[atlas] gate A (v1, loop) and B (v3 row split, stacked), $GATE batches"
python3 scripts/run_atlas.py $FROZEN --out $OUT/gate_A --stats-impl v1 --merge-impl loop --stop-after $GATE --ckpt-every $GATE --pair-every 1 || echo "[atlas] gate A rc=$?"
[ "$SMOKE" = 1 ] && export TRITON_INTERPRET=1
python3 scripts/run_atlas.py $FROZEN --out $OUT/gate_B --stats-impl v3 --merge-impl stacked --stop-after $GATE --ckpt-every $GATE --pair-every 1 || echo "[atlas] gate B rc=$?"
unset TRITON_INTERPRET
python3 scripts/compare_atlas.py $OUT/gate_A/ckpt/ckpt_$GATE.pt $OUT/gate_B/ckpt/ckpt_$GATE.pt --out $OUT/gate_compare.json > $OUT/gate_compare.log 2>&1; gate=$?
echo "[atlas] gate rc=$gate $(python3 -c "import json; r=json.load(open('$OUT/gate_compare.json')); print('ok', r['ok'], {k: v for k, v in r['fields'].items() if not (v.get('exact') or v.get('equal'))})" 2>&1 | cut -c1-500)"
if [ $gate != 0 ]; then echo "[atlas] equivalence gate FAILED: full pass not started"; kill $SMI_PID $PS_PID 2>/dev/null; exit 6; fi

echo "[atlas] full pass (prefetch 2, stats v3 with row split, merge stacked)"
t0=$(date +%s)
python3 scripts/run_atlas.py $FROZEN --out $OUT/full $MAXB --pair-every $PAIR --ckpt-every $CKPT --gram-flush-every $FLUSH --profile-at $PROF; rc=$?
echo "[atlas] full pass rc=$rc wall $(( $(date +%s) - t0 ))s"
[ -n "$SMI_PID" ] && echo "[atlas] ps nvidia-smi after pass: $(ps -o pid,etime,stat,cmd -p $SMI_PID | tail -1) | smi.csv lines $(wc -l < $OUT/smi.csv)"
python3 - <<EOF 2>&1 | sed 's/^/[atlas] /'
import json, statistics as st
R = [json.loads(l) for l in open("$OUT/full/batches.jsonl")]
j = json.load(open("$OUT/full/run.json"))
def ms(x): return f"mean {st.mean(x):.3f} median {st.median(x):.3f} (n={len(x)})" if x else "n=0"
p = [r for r in R if r.get("fwd_only_s") is not None]
c = [r for r in p if not (r["redo_layers"] or r["profiled"] or r.get("gram_flushed") or r.get("ckpt_s"))]
o = lambda lo, hi: [r["overhead_vs_fwd"] for r in c if lo <= r["batch"] < hi]
n = len(R)
print(f"batches {n} of {j['n_batches_plan']}, complete {j['complete']}; tokens {j['n_seen']:,}")
print(f"seconds: load {j['seconds']['load']:.0f}, pilot {j['seconds']['pilot']:.0f}, stream {j['seconds']['stream_this_process']:.0f}; final save {j.get('final_save_s')}")
print(f"stream wall (first batch start to last batch end) {(R[-1]['t_end'] - R[0]['t_start']) / 60:.1f} min")
print(f"forward {st.mean(r['fwd_only_s'] for r in p):.3f}s; overhead all {ms(o(0, 10**9))}; first 100 {ms(o(0, 100))}; last 1600 {ms(o(n - 1600, n))}")
g = [R[i + 1]["t_start"] - R[i]["t_end"] for i in range(n - 1) if not R[i].get("ckpt_s")]
print(f"gap between batches (excl. checkpoints) mean {st.mean(g):.3f}s, total {sum(g) / 60:.1f} min; checkpoints {sum(r['ckpt_s'] for r in R if r.get('ckpt_s')) / 60:.1f} min; paired forwards {sum(r['fwd_only_s'] for r in p) / 60:.1f} min")
print(f"redo batches {sum(1 for r in R if r['redo_layers'])}; seed verification {j.get('res_seed_limited_bins')} of {j.get('res_seeded_bins')} short")
print(f"digests {json.dumps(j.get('digests'))}")
EOF
echo "[atlas] held-out atlas (same bin edges)"
python3 scripts/run_atlas.py --model $MODEL --data $DATA --device $DEV --dtype $DT --backend $BACKEND --tokens-per-batch $TPB \
  --K 64 --Kr 64 --only-acc $HELD --edges-from $OUT/full/atlas.pt --deadline $DEADLINE --prefetch 2 --out $OUT/heldout $HMAXB \
  --ckpt-every 100000 2>&1 | grep -E "^\[atlas\]|Error|Traceback" | tail -4
echo "[atlas] redundancy per bin"
python3 scripts/bin_clusters.py --atlas $OUT/full/atlas.pt --data $DATA --uniref $UNI --out $OUT/bin_clusters.json --counts-out $OUT/bin_clusters_counts.pt 2>&1 | grep -E "^res|Error|Traceback" | sed 's/^/[atlas] /'

echo "[atlas] AlphaFold DB descriptors ($(left)s left)"
if [ "$SMOKE" = 1 ]; then
  python3 scripts/afdb_descriptors.py --data $DATA $NDESC --out $OUT/descriptors.pt 2>&1 | tail -1 | sed 's/^/[atlas] descriptors: /'
else
  AF=$HOME/afdb; mkdir -p $AF/pdb
  URL=""
  for v in 6 5 4; do
    u="https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v$v.tar"
    if curl -sfI "$u" > /dev/null; then URL=$u; break; fi
  done
  echo "[atlas] AFDB bulk: ${URL:-none found}"
  if [ -n "$URL" ]; then
    t1=$(date +%s)
    curl -sf "$URL" | tar -x -C $AF/pdb --wildcards '*-F1-model_v*.pdb.gz'
    echo "[atlas] AFDB bulk extracted $(ls $AF/pdb | wc -l) F1 models in $(( $(date +%s) - t1 ))s"
    timeout $(( $(left) - 600 )) python3 scripts/afdb_descriptors.py --data $DATA --pdb-dir $AF/pdb --procs $NPROC --out $FS/descriptors_swissprot_2026_03.pt \
      2>&1 | tail -2 | sed 's/^/[atlas] descriptors: /'
  else
    timeout $(( $(left) - 600 )) python3 scripts/afdb_descriptors.py --data $DATA --threads 64 --out $FS/descriptors_swissprot_2026_03.pt \
      2>&1 | tail -2 | sed 's/^/[atlas] descriptors (API): /'
  fi
  cp $FS/descriptors_swissprot_2026_03.pt $OUT/ 2>/dev/null
fi
[ -n "$SMI_PID" ] && echo "[atlas] ps nvidia-smi at end: $(ps -o pid,etime,stat,cmd -p $SMI_PID | tail -1) | smi.csv lines $(wc -l < $OUT/smi.csv) | smi.err: $(head -c 300 $OUT/smi.err)"
kill $SMI_PID $PS_PID 2>/dev/null
echo "[atlas] end $(date -u +%FT%TZ) rc=$rc"
exit $rc
