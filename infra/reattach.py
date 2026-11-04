"""Re-attach to a running job after the launching driver died (e.g. the controlling container
restarted): mirror its output, terminate on done.json or at the ORIGINAL hard cap, and on any
error. The instance's own runner keeps going regardless; this only restores supervision.

  python infra/reattach.py --run <run-id> --instance-id <id> --cap-at <epoch> --mirror f1 f2 ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import lambda_api as L  # noqa: E402

FS_JUP = "esm-atlas-results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--cap-at", type=float, required=True, help="epoch seconds: terminate no later than this")
    ap.add_argument("--mirror", nargs="*", default=[])
    ap.add_argument("--fs", default=None, help="filesystem name (default: the per-region name lambda_job.py uses)")
    a = ap.parse_args()
    out = Path(__file__).parent / "runs" / a.run
    out.mkdir(parents=True, exist_ok=True)
    tl_path = out / "timeline.json"
    tl = json.loads(tl_path.read_text()) if tl_path.exists() else {"run": a.run, "events": []}

    def ev(name, **kw):
        tl["events"].append({"t": L.now(), "event": name, **kw})
        print(f"[{time.strftime('%H:%M:%S')}] {name} {kw if kw else ''}", flush=True)
        tl_path.write_text(json.dumps(tl, indent=1))

    iid = a.instance_id
    ev("reattached", instance_id=iid, cap_at=a.cap_at)
    try:
        inst = L.get_instance(iid)
        if inst["status"] in ("terminated", "terminating"):
            ev("already_down", status=inst["status"])
            return
        jup = L.Jupyter(inst["jupyter_url"], inst["jupyter_token"])
        fs = a.fs or (FS_JUP if inst.get("region", {}).get("name") == "us-east-1" else f"esm-atlas-{inst['region']['name']}")
        base = f"{fs}/runs/{a.run}"
        seen = 0
        while L.now() < a.cap_at:
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
                    (out / f).parent.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
