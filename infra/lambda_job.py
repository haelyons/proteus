"""Run one batch job on a fresh Lambda instance, over HTTPS only.

  submit : cloud-init user_data starts a tiny inbox runner (as ubuntu); we PUT <job>.tar.gz
           (code + run.sh + deadline.txt) then <job>.ready into ~/jobs/inbox via Jupyter.
  output : everything the job writes goes to the persistent filesystem under runs/<job>/
           (job.log, done.json, stage files). Nothing of value lives on the instance disk.
  mirror : every poll copies job.log and --mirror files to infra/runs/<run>/ locally.
  stop   : this loop terminates on done.json or at --hard-cap seconds after launch, and in a
           finally-block on any error. The job gets DEADLINE = launch + hard_cap - 300 s so it can
           skip stages it cannot finish. Independently, the caller arms tag-scoped one-shot
           scheduled tasks (the dead-man's switch) before calling this.
"""
from __future__ import annotations

import argparse
import os
import io
import json
import sys
import tarfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import lambda_api as L  # noqa: E402

FS_MOUNT = "/lambda/nfs/esm-atlas-results"
FS_JUP = "esm-atlas-results"  # the same filesystem as seen from the Jupyter root

RUNNER = r"""#!/bin/bash
# inbox runner (as ubuntu): runs ~/jobs/inbox/<job>.tar.gz once <job>.ready exists.
J=/home/ubuntu/jobs; FS=__FS__; mkdir -p $J/inbox $J/work
while true; do
  date +%s > $J/heartbeat
  for r in $(ls $J/inbox/*.ready 2>/dev/null | sort); do
    job=$(basename $r .ready); rm -f $r
    OUT=$FS/runs/$job; mkdir -p $J/work/$job $OUT
    tar -xzf $J/inbox/$job.tar.gz -C $J/work/$job
    echo "{\"start\": $(date +%s.%N)}" > $OUT/started.json
    ( cd $J/work/$job && JOB=$job OUT=$OUT FS=$FS DEADLINE=$(cat deadline.txt 2>/dev/null || echo 0) bash run.sh ) > $OUT/job.log 2>&1
    rc=$?
    echo "{\"rc\": $rc, \"end\": $(date +%s.%N)}" > $OUT/done.json
  done
  sleep 5
done
"""

BOOT = r"""#!/bin/bash
set -x
M=/home/ubuntu/atlas_marker; mkdir -p $M
printf '{"run":"%s","runcmd_start_epoch":%s,"uptime_s_at_runcmd":%s}\n' "__RUN__" "$(date +%s.%N)" "$(cut -d' ' -f1 /proc/uptime)" > $M/marker.json
FS=__FS__
mkdir -p $FS/cache $FS/runs $FS/code && chown ubuntu:ubuntu $FS/cache $FS/runs $FS/code
chown -R ubuntu:ubuntu $M
install -o ubuntu -g ubuntu -m 0755 /usr/local/bin/atlas_runner.sh /home/ubuntu/atlas_runner.sh
sudo -u ubuntu -H bash -c 'setsid nohup /home/ubuntu/atlas_runner.sh > /home/ubuntu/atlas_runner.out 2>&1 &'
echo "boot done $(date -u +%FT%T.%NZ)"
"""


def user_data(run: str) -> str:
    def block(sh):
        return "\n".join("      " + ln for ln in sh.replace("__FS__", FS_MOUNT).splitlines())
    return ("#cloud-config\nwrite_files:\n"
            "  - path: /usr/local/bin/atlas_runner.sh\n    permissions: '0755'\n    content: |\n" + block(RUNNER) + "\n"
            "  - path: /usr/local/bin/atlas_boot.sh\n    permissions: '0755'\n    content: |\n" + block(BOOT.replace("__RUN__", run)) + "\n"
            "runcmd:\n  - [/usr/local/bin/atlas_boot.sh]\n")


def pack(job_dir: Path, include: list[Path], extra: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in sorted(job_dir.rglob("*")):
            if f.is_file():
                tf.add(f, arcname=str(f.relative_to(job_dir)))
        for p in include:
            for f in sorted(p.rglob("*")) if p.is_dir() else [p]:
                if f.is_file() and "__pycache__" not in f.parts and f.suffix not in (".pt", ".log"):
                    tf.add(f, arcname=str(Path(p.name) / f.relative_to(p)) if p.is_dir() else f.name)
        for name, text in extra.items():
            data = text.encode()
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--include", nargs="*", default=[])
    ap.add_argument("--type", default="gpu_1x_a10")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--ssh-key", default=os.environ.get("LAMBDA_SSH_KEY_NAME"),
                    help="name of an SSH key registered in the Lambda account (or set LAMBDA_SSH_KEY_NAME); "
                         "Lambda requires exactly one at launch")
    ap.add_argument("--fs", default="esm-atlas-results")
    ap.add_argument("--hard-cap", type=float, required=True, help="seconds after launch")
    ap.add_argument("--upload", nargs=2, action="append", default=[], metavar=("LOCAL", "REMOTE_UNDER_FS"),
                    help="chunked-upload LOCAL to <fs>/REMOTE if absent (e.g. cache/swissprot.pt)")
    ap.add_argument("--mirror", nargs="*", default=[], help="files under runs/<job>/ to mirror locally each poll")
    a = ap.parse_args()
    if not a.ssh_key:
        ap.error("--ssh-key or LAMBDA_SSH_KEY_NAME is required")
    out = Path(__file__).parent / "runs" / a.run
    out.mkdir(parents=True, exist_ok=True)
    tl = {"run": a.run, "type": a.type, "region": a.region, "events": []}

    def ev(name, **kw):
        tl["events"].append({"t": L.now(), "event": name, **kw})
        print(f"[{time.strftime('%H:%M:%S')}] {name} {kw if kw else ''}", flush=True)
        (out / "timeline.json").write_text(json.dumps(tl, indent=1))

    iid, job = None, a.run
    try:
        t_launch = L.now()
        iid = L.launch(region_name=a.region, instance_type_name=a.type, ssh_key_names=[a.ssh_key],
                       file_system_names=[a.fs], name=a.run, user_data=user_data(a.run),
                       tags=[{"key": "atlas-run", "value": a.run}])
        ev("launched", instance_id=iid)
        cap_at = t_launch + a.hard_cap
        inst, last = None, None
        while L.now() < cap_at:
            inst = L.get_instance(iid)
            if inst["status"] != last:
                ev("status", status=inst["status"], first_healthy=inst.get("first_healthy"))
                last = inst["status"]
            if inst["status"] == "active" and inst.get("jupyter_url") and inst.get("jupyter_token"):
                break
            if inst["status"] in ("terminated", "terminating", "unhealthy"):
                raise RuntimeError(f"instance {inst['status']} before active")
            time.sleep(5)
        jup = L.Jupyter(inst["jupyter_url"], inst["jupyter_token"])
        while L.now() < cap_at:
            try:
                jup.read_text("jobs/heartbeat")
                break
            except Exception:  # noqa: BLE001
                time.sleep(5)
        ev("runner_alive")
        for local, remote in a.upload:
            rp = f"{FS_JUP}/{remote}"
            try:
                _, meta = jup.get(rp, content=False)
                ev("upload_skipped_exists", remote=rp, size=meta.get("size"))
            except Exception:  # noqa: BLE001
                size, secs = L.put_file_chunked(jup, rp, local, log=lambda m: ev("upload_progress", msg=m))
                ev("uploaded", remote=rp, bytes=size, seconds=round(secs, 1), MBps=round(size / secs / 1e6, 1))
        deadline = cap_at - 300
        blob = pack(Path(a.job_dir), [Path(p) for p in a.include], {"deadline.txt": f"{deadline:.0f}\n"})
        jup.put_b64(f"jobs/inbox/{job}.tar.gz", blob)
        jup.put_text(f"jobs/inbox/{job}.ready", "")
        ev("submitted", bytes=len(blob), job_deadline=deadline)
        base = f"{FS_JUP}/runs/{job}"
        seen = 0
        while L.now() < cap_at:
            try:
                log = jup.read_text(f"{base}/job.log")
                (out / "job.log").write_text(log)
                for ln in log[seen:].splitlines():
                    if ln.startswith("[atlas]") or "Error" in ln or "Traceback" in ln:
                        print("   |", ln[:400], flush=True)
                seen = len(log)
            except Exception:  # noqa: BLE001
                pass
            for f in a.mirror:
                try:
                    (out / f).write_text(jup.read_text(f"{base}/{f}"))
                except Exception:  # noqa: BLE001
                    pass
            try:
                done = json.loads(jup.read_text(f"{base}/done.json"))
                ev("job_done", **done)
                break
            except Exception:  # noqa: BLE001
                time.sleep(20)
        else:
            ev("hard_cap_reached")
    except Exception as e:  # noqa: BLE001
        ev("error", error="".join(traceback.format_exception_only(type(e), e)).strip())
    finally:
        if iid:
            try:
                st = L.get_instance(iid)["status"]
            except Exception:  # noqa: BLE001
                st = "unknown"
            if st not in ("terminated", "terminating"):
                L.terminate([iid])
                ev("terminate_called", prior_status=st)
            for _ in range(30):
                try:
                    st = L.get_instance(iid)["status"]
                except Exception:  # noqa: BLE001
                    pass
                if st == "terminated":
                    break
                time.sleep(10)
            ev("final_status", status=st)
        (out / "timeline.json").write_text(json.dumps(tl, indent=1))


if __name__ == "__main__":
    main()
