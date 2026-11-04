"""Attribute device time to profiler ranges from a Chrome trace (torch.profiler export).

key_averages() gives each record_function range a "CUDA total" that is the span of its GPU
annotation: it includes idle gaps (launch latency, host syncs) and cannot say which kernels
inside it took the time. Here every kernel / memcpy / memset is assigned to its innermost
enclosing GPU annotation, so per range we report:
  span   - sum of annotation durations (what key_averages calls CUDA total)
  busy   - device time of the kernels inside it (all nesting levels)
  idle   - span - busy (gaps: syncs, launch-bound stretches)
and the top kernels by innermost range. On CPU-only traces the same is done with cpu ops whose
immediate parent is a user annotation (smoke tests).

  python -m atlas.proftrace trace.json[.gz] [--top 8]
"""
from __future__ import annotations

import collections
import gzip
import json
import sys

DEV_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def _load(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        tr = json.load(f)
    return tr["traceEvents"] if isinstance(tr, dict) else tr


def attribute(path, prefix="atlas.") -> dict:
    ev = [e for e in _load(path) if e.get("ph") == "X" and "dur" in e]
    dev = [e for e in ev if e.get("cat") in DEV_CATS]
    if dev:
        ann = [e for e in ev if e.get("cat") == "gpu_user_annotation" and e["name"].startswith(prefix)]
        work = dev
        key = lambda e: e.get("pid")  # noqa: E731  annotations and kernels share the device pid
    else:  # CPU trace: ops nested in annotations; count ops whose parent is an annotation
        ann = [e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith(prefix)]
        work = []
        ann_ids = {id(a) for a in ann}
        per_tid = collections.defaultdict(list)
        for e in ev:
            if e.get("cat") == "cpu_op" or id(e) in ann_ids:
                per_tid[e.get("tid")].append(e)
        for es in per_tid.values():  # stack nesting: keep ops whose immediate parent is an annotation
            es.sort(key=lambda e: (e["ts"], -e["dur"]))
            stack = []
            for e in es:
                while stack and e["ts"] >= stack[-1]["ts"] + stack[-1]["dur"]:
                    stack.pop()
                if e.get("cat") == "cpu_op" and stack and stack[-1].get("cat") == "user_annotation":
                    work.append(e)
                stack.append(e)
        key = lambda e: e.get("tid")  # noqa: E731
    by = collections.defaultdict(list)
    for a in ann:
        by[key(a)].append(a)
    span = collections.Counter()
    for a in ann:
        span[a["name"]] += a["dur"]
    busy = collections.Counter()
    self_k = collections.defaultdict(collections.Counter)
    calls = collections.defaultdict(collections.Counter)
    total = 0.0
    for k in work:
        total += k["dur"]
        mid = k["ts"] + k["dur"] / 2
        enc = [a for a in by.get(key(k), []) if a["ts"] <= mid <= a["ts"] + a["dur"]]
        if not enc:
            continue
        enc.sort(key=lambda a: a["dur"])
        for name in {a["name"] for a in enc}:
            busy[name] += k["dur"]
        nm = k["name"]
        self_k[enc[0]["name"]][nm] += k["dur"]
        calls[enc[0]["name"]][nm] += 1
    return {"span_us": dict(span), "busy_us": dict(busy), "device_total_us": total,
            "self_kernels": {r: [(n, round(us, 1), calls[r][n]) for n, us in c.most_common()] for r, c in self_k.items()},
            "mode": "device" if dev else "cpu"}


def table(res, top=8) -> str:
    out = [f"mode {res['mode']}; device busy total {res['device_total_us'] / 1e3:.1f} ms",
           f"{'range':28s} {'span ms':>9s} {'busy ms':>9s} {'idle ms':>9s}"]
    for name in sorted(res["span_us"]):
        sp, bu = res["span_us"][name] / 1e3, res["busy_us"].get(name, 0.0) / 1e3
        out.append(f"{name:28s} {sp:9.2f} {bu:9.2f} {sp - bu:9.2f}")
    for name in sorted(res["self_kernels"]):
        ks = res["self_kernels"][name]
        out.append(f"\n[{name}] self kernels (top {top} of {len(ks)}): ms, calls")
        for n, us, c in ks[:top]:
            out.append(f"  {us / 1e3:8.2f} {c:5d}  {n[:150]}")
    return "\n".join(out)


if __name__ == "__main__":
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 8
    print(table(attribute(sys.argv[1]), top))
