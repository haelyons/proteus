"""Access test: user_data writes a marker; read it back through Jupyter's contents API.

Timeline recorded (local wall clock, UTC epoch seconds):
  launch_req -> status transitions (5 s poll) -> first_healthy (API field)
  -> first successful Jupyter contents GET -> marker read -> PUT round-trip
  -> dead-man's switch window -> terminated.

Safety: any exception, or the hard cap, terminates the instance. In the
dead-man phase this loop deliberately does NOT terminate until
`deadman_at + grace`, so the external scheduled task is what is being tested;
if it has not acted by then, this loop terminates and records the failure.
"""
from __future__ import annotations

import argparse
import os
import json
import socket
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import lambda_api as L  # noqa: E402

BOOT_SH = r"""#!/bin/bash
RUN=__RUN__
M=/home/ubuntu/atlas_marker
mkdir -p $M
T0=$(date +%s.%N); UP=$(cut -d' ' -f1 /proc/uptime); BOOT=$(date -d "$(uptime -s)" +%s)
printf '{"run":"%s","runcmd_start_epoch":%s,"uptime_s_at_runcmd":%s,"boot_epoch":%s,"hostname":"%s"}\n' "$RUN" "$T0" "$UP" "$BOOT" "$(hostname)" > $M/marker.json
FS=/lambda/nfs/esm-atlas-results
ls -la /lambda/nfs > $M/nfs_ls.txt 2>&1
if [ -d "$FS" ]; then mkdir -p $FS/markers && cp $M/marker.json $FS/markers/$RUN.json && chown -R ubuntu:ubuntu $FS/markers && ln -sfn $FS /home/ubuntu/results; fi
chown -R ubuntu:ubuntu $M; chown -h ubuntu:ubuntu /home/ubuntu/results 2>/dev/null
setsid nohup bash -c '
  M=/home/ubuntu/atlas_marker
  { nvidia-smi; echo; python3 -c "import sys,torch; print(sys.version); print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, torch.cuda.is_available())";
    python3 -c "import triton; print(\"triton\", triton.__version__)";
    python3 -c "import transformers; print(\"transformers\", transformers.__version__)";
    df -h /home /lambda/nfs/*; nproc; free -g; } > $M/env.txt 2>&1
  for i in $(seq 1 150); do cp /var/log/cloud-init-output.log $M/cloud-init-output.log 2>/dev/null;
    cloud-init status --long > $M/cloud-init-status.txt 2>&1; date +%s.%N > $M/heartbeat.txt;
    chown -R ubuntu:ubuntu $M; sleep 10; done' >/dev/null 2>&1 &
echo "atlas_boot runcmd finished at $(date -u +%FT%T.%NZ)"
"""


def user_data(run: str) -> str:
    sh = BOOT_SH.replace("__RUN__", run)
    body = "\n".join("      " + ln for ln in sh.splitlines())
    return (
        "#cloud-config\n"
        "write_files:\n"
        "  - path: /usr/local/bin/atlas_boot.sh\n"
        "    permissions: '0755'\n"
        "    content: |\n" + body + "\n"
        "runcmd:\n"
        "  - [/usr/local/bin/atlas_boot.sh]\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--type", default="gpu_1x_a10")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--ssh-key", default=os.environ.get("LAMBDA_SSH_KEY_NAME"),
                    help="name of an SSH key registered in the Lambda account (or set LAMBDA_SSH_KEY_NAME); "
                         "Lambda requires exactly one at launch")
    ap.add_argument("--fs", default="esm-atlas-results")
    ap.add_argument("--deadman-at", type=float, required=True, help="epoch when the scheduled task fires")
    ap.add_argument("--grace", type=float, default=12 * 60)
    ap.add_argument("--hard-cap", type=float, default=55 * 60, help="seconds after launch")
    a = ap.parse_args()
    if not a.ssh_key:
        ap.error("--ssh-key or LAMBDA_SSH_KEY_NAME is required")

    out = Path(__file__).parent / "runs" / a.run
    out.mkdir(parents=True, exist_ok=True)
    tl: dict = {"run": a.run, "type": a.type, "region": a.region, "events": [], "deadman_at": a.deadman_at}

    def ev(name, **kw):
        tl["events"].append({"t": L.now(), "event": name, **kw})
        print(f"[{time.strftime('%H:%M:%S')}] {name} {kw if kw else ''}", flush=True)
        (out / "timeline.json").write_text(json.dumps(tl, indent=1))

    iid = None
    terminated_by_us = False
    try:
        tl["t_launch_req"] = L.now()
        iid = L.launch(
            region_name=a.region, instance_type_name=a.type, ssh_key_names=[a.ssh_key],
            file_system_names=[a.fs], name=a.run, user_data=user_data(a.run),
            tags=[{"key": "atlas-run", "value": a.run}],
        )
        tl["instance_id"] = iid
        ev("launched", instance_id=iid)
        t_cap = tl["t_launch_req"] + a.hard_cap

        # ---- phase 1: status until active
        last = None
        inst = None
        while L.now() < t_cap:
            inst = L.get_instance(iid)
            st = inst.get("status")
            if st != last:
                ev("status", status=st, first_healthy=inst.get("first_healthy"),
                   jupyter_url=L.redact(inst.get("jupyter_url")), hostname=inst.get("hostname"), ip=inst.get("ip"))
                last = st
            if st == "active":
                break
            if st in ("terminated", "terminating", "unhealthy"):
                raise RuntimeError(f"instance went {st} before active")
            time.sleep(5)
        tl["t_active_seen"] = L.now()

        # ---- phase 2: Jupyter reachability (refresh object until URL+token present)
        while not (inst.get("jupyter_url") and inst.get("jupyter_token")) and L.now() < t_cap:
            time.sleep(5)
            inst = L.get_instance(iid)
        jup = L.Jupyter(inst["jupyter_url"], inst["jupyter_token"])
        tl["jupyter_host"] = jup.host
        try:
            tl["jupyter_host_dns"] = sorted({ai[4][0] for ai in socket.getaddrinfo(jup.host, 443)})
        except Exception as e:  # noqa: BLE001
            tl["jupyter_host_dns"] = f"no resolution: {e}"
        ev("jupyter_url_known", host=jup.host, dns=tl["jupyter_host_dns"])

        errs: dict[str, int] = {}
        first_ok = None
        while L.now() < t_cap:
            try:
                code, root = jup.get("", content=True)
                first_ok = L.now()
                tl["jupyter_root_listing"] = [c["name"] for c in root.get("content", [])]
                ev("jupyter_reachable", http=code, root=tl["jupyter_root_listing"], errors_before=errs)
                break
            except Exception as e:  # noqa: BLE001
                k = L.classify_error(e)
                if k not in errs:
                    ev("jupyter_attempt_error", error=k)
                errs[k] = errs.get(k, 0) + 1
                if L.now() - tl["t_active_seen"] > 15 * 60:
                    raise RuntimeError(f"Jupyter unreachable 15 min after active: {errs}")
                time.sleep(5)
        tl["t_jupyter_ok"] = first_ok

        # ---- phase 3: marker + artefacts
        marker = None
        while marker is None and L.now() < t_cap:
            try:
                marker = json.loads(jup.read_text("atlas_marker/marker.json"))
            except Exception as e:  # noqa: BLE001
                ev("marker_not_yet", error=L.classify_error(e))
                time.sleep(10)
        tl["marker"] = marker
        tl["t_marker_read"] = L.now()
        ev("marker_read", marker=marker)
        time.sleep(20)  # let env.txt land
        for f in ["env.txt", "nfs_ls.txt", "cloud-init-status.txt", "cloud-init-output.log"]:
            try:
                (out / f).write_text(jup.read_text(f"atlas_marker/{f}"))
                ev("fetched", file=f)
            except Exception as e:  # noqa: BLE001
                ev("fetch_failed", file=f, error=L.classify_error(e))
        try:
            fs_marker = jup.read_text(f"results/markers/{a.run}.json")
            tl["fs_marker_via_symlink"] = json.loads(fs_marker)
            ev("fs_marker_read", ok=True)
        except Exception as e:  # noqa: BLE001
            ev("fs_marker_read", ok=False, error=L.classify_error(e))
        # write channel: PUT then read back (into the persistent filesystem via the symlink)
        probe = json.dumps({"put_at": L.now(), "run": a.run})
        for path in ["atlas_marker/put_probe.json", "results/markers/put_probe.json"]:
            try:
                t0 = L.now()
                jup.put_text(path, probe)
                back = jup.read_text(path)
                ev("put_roundtrip", path=path, ok=(back == probe), seconds=round(L.now() - t0, 2))
            except Exception as e:  # noqa: BLE001
                ev("put_roundtrip", path=path, ok=False, error=L.classify_error(e))

        # ---- phase 4: dead-man's switch window. We do not terminate; we watch.
        ev("deadman_wait", fires_at=a.deadman_at, our_fallback_at=a.deadman_at + a.grace)
        seen = None
        while L.now() < min(a.deadman_at + a.grace, t_cap):
            st = L.get_instance(iid).get("status")
            if st != seen:
                ev("status", status=st)
                seen = st
            if st in ("terminating", "terminated"):
                tl["deadman_worked"] = True
                break
            # keep refreshing the cloud-init log while we wait (cheap)
            if int(L.now()) % 120 < 20:
                try:
                    (out / "cloud-init-output.log").write_text(jup.read_text("atlas_marker/cloud-init-output.log"))
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(20)
        else:
            tl["deadman_worked"] = False
    except Exception as e:  # noqa: BLE001
        tl["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()
        ev("error", error=tl["error"])
    finally:
        if iid:
            try:
                st = L.get_instance(iid).get("status")
            except Exception:  # noqa: BLE001
                st = "unknown"
            if st not in ("terminating", "terminated"):
                L.terminate([iid])
                terminated_by_us = True
                ev("terminate_called_by_loop", prior_status=st)
            t_end = L.now() + 300
            while L.now() < t_end:
                st = L.get_instance(iid).get("status")
                if st == "terminated":
                    break
                time.sleep(10)
            ev("final_status", status=st, terminated_by_loop=terminated_by_us)
        tl["t_end"] = L.now()
        (out / "timeline.json").write_text(json.dumps(tl, indent=1))


if __name__ == "__main__":
    main()
