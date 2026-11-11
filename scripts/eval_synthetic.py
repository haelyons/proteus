"""Unblind ruler calls on synthetic columns and score them against decision rule v1's gate.

  python scripts/eval_synthetic.py --calls calls.json --columns synth/columns.json --out gate.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

NONE = "no association found by this procedure"
EXPECT = {  # (call, primary descriptor, mode label) expected at the stated SNR or above; persistence where it is part of the gate
    "helix": {"call": {"structural association"}, "primary": "state5", "modes": "one mode", "top": 1},
    "buried": {"call": {"structural association"}, "primary": "burial3", "modes": "one mode", "top": 2},
    "trp": {"call": {"composition association"}},
    "lys_helix": {"call": {"context association"}},
    "two_job": {"call": {"structural association", "context association"}, "modes": "several modes"},
    "ordered": {"call": {"structural association", "context association"}, "modes": "several modes"},
    "narrow": {"call": {"structural association", "context association", "composition association"}, "persistence": "peak only"},
    "hydro_win": {"call": {"structural association", "context association", "composition association"}},
    "noise": {"call": {NONE}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True)
    ap.add_argument("--columns", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-alpha", type=float, default=1.0)
    a = ap.parse_args()
    calls = {c["col"]: c for c in json.load(open(a.calls))["calls"]}
    cols = json.load(open(a.columns))
    table = collections.defaultdict(lambda: collections.Counter())
    gate = collections.defaultdict(lambda: [0, 0])
    sub = collections.defaultdict(lambda: [0, 0])
    ungraded = collections.Counter()
    for col in cols:
        c = calls[col["col"]]; t, al = col["type"], col["alpha"]
        key = c["call"] + (f" / {c.get('primary')} {c.get('mode_label', '')} {c.get('persistence', '')}".rstrip() if c.get("mode_label") else "")
        table[(t, al)][key] += 1
        if t not in EXPECT:
            ungraded[(t, al)] += 1
            continue
        if 0 < al < a.min_alpha and t != "noise":  # below the gate's SNR: descriptive only (detected = any association)
            sub[t][0] += c["call"] != NONE; sub[t][1] += 1
            continue
        e = EXPECT[t] if al >= a.min_alpha or t == "noise" else EXPECT["noise"]   # alpha = 0: pure noise
        ok = c["call"] in e["call"]
        if ok and "primary" in e:
            ok = c.get("primary") == e["primary"] and c.get("top_mode") == e["top"]
        if ok and "modes" in e:
            ok = c.get("mode_label") == e["modes"]
        if ok and "persistence" in e:
            ok = c.get("persistence") == e["persistence"]
        g = t if (al >= a.min_alpha and t != "noise") else "noise + every alpha = 0"
        gate[g][0] += ok; gate[g][1] += 1
    rows = {f"{t} alpha={al}": dict(v) for (t, al), v in sorted(table.items())}
    res = {"min_alpha": a.min_alpha, "gate": {k: f"{v[0]}/{v[1]}" for k, v in sorted(gate.items())},
           "detected_below_gate": {k: f"{v[0]}/{v[1]}" for k, v in sorted(sub.items())}, "calls_by_type_alpha": rows}
    if ungraded:
        res["ungraded"] = {f"{t} alpha={al}": n for (t, al), n in sorted(ungraded.items())}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for k, v in sorted(gate.items()):
        print(f"{k:30s} {v[0]}/{v[1]}")
    for k, v in sorted(sub.items()):
        print(f"{k:16s} alpha in (0, {a.min_alpha}): any association {v[0]}/{v[1]}")
    for k, v in rows.items():
        print(f"  {k:22s} {v}")


if __name__ == "__main__":
    main()
