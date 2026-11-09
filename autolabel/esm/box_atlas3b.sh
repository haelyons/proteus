#!/bin/bash
# Launch the ESM-2 3B atlas job (atlas3b_job.sh) on one A100 40 GB in us-east-1 with esm-atlas-results attached, then
# supervise it: poll every 5 min, fetch the bundle and logs at DONE or FAILED, terminate. Meant to run detached:
#   setsid nohup bash box_atlas3b.sh > box/atlas3b/supervisor.log 2>&1 &
# A separate detached watchdog terminates the instance after CAP_S seconds. The job writes to the filesystem under
# runs/autolabel-atlas3b-<stamp>/ and gets DEADLINE = launch + CAP_S - 1200 so it can checkpoint and stop cleanly.
set -uo pipefail
HERE="$HOME/dev/interp/autolabel/esm"; BOX="$HERE/box/atlas3b"; DATA="$HERE/data"; mkdir -p "$BOX" "$DATA"
KEYS="$HOME/dev/interp/dm_hack/.keys"; API=https://cloud.lambda.ai/api/v1
TYPE=${TYPE:-gpu_1x_a100_sxm4}; REGION=us-east-1; FSN=esm-atlas-results; SSHKEY=latent_verify_hal_20260721; CAP_S=${CAP_S:-36000}
FS=/lambda/nfs/$FSN; STAMP=$(date -u +%Y%m%d-%H%M); OUT=$FS/runs/autolabel-atlas3b-$STAMP
SSHO="-i $HOME/.ssh/lambda_ed25519 -o UserKnownHostsFile=$BOX/known_hosts -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30"
key() { grep -E '^LAMBDA_KEY_ONE=' "$KEYS" | cut -d= -f2- | tr -d '\r\n'; }
api() { curl -s --max-time 60 -u "$(key):" -H 'Content-Type: application/json' "$@"; }
say() { echo "[3b-sup $(date -u +%H:%M:%S)] $*"; }
terminate() { api -X POST "$API/instance-operations/terminate" -d "{\"instance_ids\":[\"$1\"]}" | python3 -I -c 'import json,sys; d=json.load(sys.stdin); print("terminate:", [i.get("status") for i in d.get("data",{}).get("terminated_instances",[])] or d.get("error"))'; }
status() { api "$API/instances/$1" | python3 -I -c 'import json,sys; d=json.load(sys.stdin).get("data",{}); print(d.get("status","gone"), d.get("ip") or "-")'; }

say "launching $TYPE in $REGION with $FSN; out $OUT"
ID=$(api -X POST "$API/instance-operations/launch" -d "{\"region_name\":\"$REGION\",\"instance_type_name\":\"$TYPE\",\"ssh_key_names\":[\"$SSHKEY\"],\"file_system_names\":[\"$FSN\"],\"name\":\"autolabel-atlas3b\",\"quantity\":1}" \
     | python3 -I -c 'import json,sys; d=json.load(sys.stdin); print((d.get("data") or {}).get("instance_ids",[""])[0] or "ERR:"+json.dumps(d.get("error")))')
case "$ID" in ERR:*|"") say "launch failed: $ID"; echo "launch-failed" > "$BOX/finished"; exit 2;; esac
T0=$(date +%s); echo "$ID" > "$BOX/instance_id"; echo "$OUT" > "$BOX/out_dir"; say "instance $ID"
setsid nohup bash -c "sleep $CAP_S; K=\$(grep -E '^LAMBDA_KEY_ONE=' '$KEYS' | cut -d= -f2- | tr -d '\r\n'); curl -s -u \"\$K:\" -H 'Content-Type: application/json' -X POST $API/instance-operations/terminate -d '{\"instance_ids\":[\"$ID\"]}' > '$BOX/watchdog.out' 2>&1" > /dev/null 2>&1 &
echo $! > "$BOX/watchdog.pid"; say "watchdog: terminate after ${CAP_S}s"
RESULT=unknown
trap 'say "exit ($RESULT): terminating $ID"; terminate "$ID"; kill -- -$(cat "$BOX/watchdog.pid") 2>/dev/null; echo "$RESULT $(( ($(date +%s) - T0) / 60 )) min" > "$BOX/finished"' EXIT

for i in $(seq 1 120); do read -r ST IP < <(status "$ID"); [ "$ST" = active ] && [ "$IP" != "-" ] && break; [ $((i % 6)) = 0 ] && say "status $ST"; sleep 10; done
[ "$ST" = active ] || { RESULT="not-active($ST)"; exit 3; }
say "active at $IP"; echo "$IP" > "$BOX/ip"
ok=0; for i in $(seq 1 60); do if ssh $SSHO ubuntu@$IP true 2>/dev/null; then ok=$((ok + 1)); else ok=0; fi; [ $ok -ge 3 ] && break; sleep 10; done
say "ssh stable after $i probes"
retry() { local n; for n in 1 2 3 4 5; do "$@" && return 0; say "retry $n: $1 failed"; sleep $((n * 15)); done; return 1; }
(cd "$HOME/dev/proteus" && git archive --format=tar.gz --prefix=proteus/ HEAD) > "$BOX/proteus.tar.gz"
retry scp -q $SSHO "$BOX/proteus.tar.gz" "$HERE/atlas3b_job.sh" "$HERE/export_firstread.py" ubuntu@$IP:~/ || { RESULT=scp-failed; exit 4; }
DEADLINE=$(( T0 + CAP_S - 1200 ))
retry ssh $SSHO ubuntu@$IP "test -f $OUT/job.log && echo already-started && exit 0; tar xzf proteus.tar.gz && mkdir -p $OUT && cd ~ && { setsid nohup env OUT=$OUT FS=$FS DEADLINE=$DEADLINE bash ~/atlas3b_job.sh > $OUT/job.log 2>&1 < /dev/null & } ; sleep 2; echo started" || { RESULT=start-failed; exit 5; }
say "job started; deadline $(date -u -d @$DEADLINE +%H:%M) UTC"
miss=0
while true; do
  sleep 300
  line=$(ssh $SSHO ubuntu@$IP "if [ -f $OUT/DONE ]; then echo DONE; elif [ -f $OUT/FAILED ]; then echo FAILED; else echo RUN \$(tail -n 1 $OUT/job.log | cut -c1-120) '|' batches \$(wc -l < $OUT/full/batches.jsonl 2>/dev/null || echo 0) '|' \$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null); fi" 2>/dev/null)
  if [ -z "$line" ]; then
    miss=$((miss + 1)); read -r ST _ < <(status "$ID"); say "ssh failed ($miss); instance $ST"
    [ "$ST" = active ] || { RESULT="instance-$ST"; exit 6; }; [ $miss -ge 6 ] && { RESULT=unreachable; exit 7; }; continue
  fi
  miss=0; say "$line"
  case "$line" in DONE|FAILED) RESULT=$line; break;; esac
done
mkdir -p "$BOX/logs"
scp -q $SSHO "ubuntu@$IP:$OUT/*.log" "ubuntu@$IP:$OUT/*.json" "ubuntu@$IP:$OUT/DONE" "ubuntu@$IP:$OUT/FAILED" "$BOX/logs/" 2>/dev/null
scp -q $SSHO "ubuntu@$IP:$OUT/full/run.json" "$BOX/logs/full_run.json" 2>/dev/null
if [ "$RESULT" = DONE ]; then
  scp -q $SSHO "ubuntu@$IP:$OUT/export/firstread_3B.pt" "$DATA/" && say "bundle fetched: $(du -h "$DATA/firstread_3B.pt" | cut -f1)" || RESULT=fetch-failed
fi
tail -n 12 "$BOX/logs/job.log"
