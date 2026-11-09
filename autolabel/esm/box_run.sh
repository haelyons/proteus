#!/bin/bash
# One Lambda A10 in us-east-1 with esm-atlas-results attached (read only): export first-read bundles, fetch, terminate.
# A detached watchdog terminates the instance after CAP_S seconds whatever happens here. Key never printed.
set -uo pipefail
HERE="$HOME/dev/interp/autolabel/esm"; BOX="$HERE/box"; DATA="$HERE/data"; mkdir -p "$BOX" "$DATA"
KEYS="$HOME/dev/interp/dm_hack/.keys"; API=https://cloud.lambda.ai/api/v1
TYPE=${TYPE:-gpu_1x_a10}; REGION=us-east-1; FSN=esm-atlas-results; SSHKEY=latent_verify_hal_20260721; CAP_S=${CAP_S:-5400}
SSH="ssh -i $HOME/.ssh/lambda_ed25519 -o UserKnownHostsFile=$BOX/known_hosts -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30"
SCP="scp -i $HOME/.ssh/lambda_ed25519 -o UserKnownHostsFile=$BOX/known_hosts -o StrictHostKeyChecking=accept-new"
key() { grep -E '^LAMBDA_KEY_ONE=' "$KEYS" | cut -d= -f2- | tr -d '\r\n'; }
api() { curl -s --max-time 60 -u "$(key):" -H 'Content-Type: application/json' "$@"; }
say() { echo "[box $(date -u +%H:%M:%S)] $*"; }
terminate() { api -X POST "$API/instance-operations/terminate" -d "{\"instance_ids\":[\"$1\"]}" | python3 -I -c 'import json,sys; d=json.load(sys.stdin); print("terminate:", [i.get("status") for i in d.get("data",{}).get("terminated_instances",[])] or d.get("error"))'; }

say "launching $TYPE in $REGION with $FSN"
ID=$(api -X POST "$API/instance-operations/launch" -d "{\"region_name\":\"$REGION\",\"instance_type_name\":\"$TYPE\",\"ssh_key_names\":[\"$SSHKEY\"],\"file_system_names\":[\"$FSN\"],\"name\":\"autolabel-esm-export\",\"quantity\":1}" \
     | python3 -I -c 'import json,sys; d=json.load(sys.stdin); print((d.get("data") or {}).get("instance_ids",[""])[0] or "ERR:"+json.dumps(d.get("error")))')
case "$ID" in ERR:*|"") say "launch failed: $ID"; exit 2;; esac
echo "$ID" > "$BOX/instance_id"; say "instance $ID"
# watchdog: detached, survives this script and the session
setsid nohup bash -c "sleep $CAP_S; K=\$(grep -E '^LAMBDA_KEY_ONE=' '$KEYS' | cut -d= -f2- | tr -d '\r\n'); curl -s -u \"\$K:\" -H 'Content-Type: application/json' -X POST $API/instance-operations/terminate -d '{\"instance_ids\":[\"$ID\"]}' > '$BOX/watchdog.out' 2>&1" > /dev/null 2>&1 &
echo $! > "$BOX/watchdog.pid"; say "watchdog armed: terminate after ${CAP_S}s (pid $(cat $BOX/watchdog.pid))"
trap 'say "exit: terminating $ID"; terminate "$ID"; kill -- -$(cat "$BOX/watchdog.pid") 2>/dev/null' EXIT   # setsid: pid = process group

IP=""; for i in $(seq 1 90); do
  read -r ST IP < <(api "$API/instances/$ID" | python3 -I -c 'import json,sys; d=json.load(sys.stdin).get("data",{}); print(d.get("status","?"), d.get("ip") or "-")')
  [ "$ST" = active ] && [ "$IP" != "-" ] && break; [ $((i % 6)) = 0 ] && say "status $ST"; sleep 10; done
[ "$ST" = active ] || { say "not active after 15 min (status $ST)"; exit 3; }
say "active at $IP"
for i in $(seq 1 30); do $SSH ubuntu@$IP true 2>/dev/null && break; sleep 10; done
$SCP -q "$HERE/export_firstread.py" ubuntu@$IP:~/ || { say "scp failed"; exit 4; }
$SSH ubuntu@$IP 'bash -s' > "$BOX/remote.log" 2>&1 <<'REMOTE'
set -uo pipefail
FS=/lambda/nfs/esm-atlas-results
A8=$FS/runs/atlas-batch-20261105-0258; A650=$FS/runs/atlas-fu650-20261104-1425/full/atlas.pt
DATA=$FS/cache/swissprot_2026_03.pt; DESC=$FS/descriptors_swissprot_2026_03.pt
echo "== inputs"; ls -la $A8/a8_pre/slim.pt $A8/a8_shuf2/slim.pt $A650 $DATA $DESC; ls $FS/cache/hf/hub 2>/dev/null | grep -i esm
nproc; free -g | head -2
python3 -m pip install --user -q -U "pillow>=10" "transformers>=4.46" 2>&1 | tail -1   # Lambda Stack's PIL breaks transformers (as in proteus jobs)
python3 -c "from transformers import EsmModel, EsmTokenizer; print('preflight: EsmModel import ok')" || { echo "PREFLIGHT FAILED"; exit 2; }
export HF_HOME=$FS/cache/hf PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$(nproc)
mkdir -p ~/out
[ "${WANT_SEQ:-0}" = 1 ] && { python3 export_firstread.py sequences --data $DATA --out ~/out || echo "SEQUENCES FAILED rc=$?"; }   # fetched by run 1
python3 export_firstread.py bundle --model 8M --atlas $A8/a8_pre/slim.pt --control $A8/a8_shuf2/slim.pt --data $DATA --desc $DESC --out ~/out --res-per-bin 16 || echo "BUNDLE 8M FAILED rc=$?"
python3 export_firstread.py bundle --model 650M --atlas $A650 --data $DATA --desc $DESC --out ~/out --res-per-bin 8 || echo "BUNDLE 650M FAILED rc=$?"
ls -la ~/out; echo REMOTE-DONE
REMOTE
grep -v -i warning "$BOX/remote.log" | tail -n 25
say "fetching bundles"
$SCP -q "ubuntu@$IP:~/out/*" "$DATA/" && ls -la "$DATA"
say "done"
