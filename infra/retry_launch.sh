#!/bin/bash
# Retry a lambda_job.py launch every INTERVAL seconds until it gets capacity (exit 4 = none, 5 = launch refused),
# for at most TRIES attempts. Once launched, lambda_job.py supervises the job to the end.
# Usage: TRIES=36 INTERVAL=300 bash infra/retry_launch.sh <lambda_job.py args...>
TRIES=${TRIES:-36}; INTERVAL=${INTERVAL:-300}
for i in $(seq 1 $TRIES); do
  echo "[retry] attempt $i/$TRIES $(date -u +%FT%TZ)"
  python3 -u infra/lambda_job.py "$@"; rc=$?
  if [ $rc -ne 4 ] && [ $rc -ne 5 ]; then echo "[retry] launcher exited rc=$rc"; exit $rc; fi  # 4: no capacity, 5: launch refused
  sleep $INTERVAL
done
echo "[retry] no capacity after $TRIES attempts; nothing launched"
