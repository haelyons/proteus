"""Collect planted-column and pooling checks from a ruler_checks job directory."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.compare import share_ci  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def shape_report(root):
    columns = {c["col"]: c for c in read(root / "columns.json")}
    calls = {c["col"]: c for c in read(root / "calls.json")["calls"]}
    rows = records(root / "scores" / f"{root.name}.jsonl")
    if set(columns) != set(calls) or set(columns) != {r["unit"] for r in rows}:
        raise ValueError(f"incomplete planted columns: {root}")
    groups = defaultdict(list)
    noise = []
    for r in rows:
        c = columns[r["unit"]]
        groups[f"{c['type']}@{c['alpha']}"].append(r)
        if c["type"] == "noise":
            noise.append(r)
    by_type = {}
    for key, group in sorted(groups.items()):
        jsd = [r["primary_jsd"] for r in group if r["primary_jsd"] is not None]
        modes = Counter(aa for r in group for aa in (r["residue_modes"] or []))
        by_type[key] = {"calls": dict(Counter(calls[r["unit"]]["call"] for r in group)),
                        "n": len(group), "k_struct": sum(structural(r) for r in group),
                        "jsd_q50": statistics.median(jsd) if jsd else None, "residue_modes": dict(modes)}
    k, n = sum(structural(r) for r in noise), len(noise)
    return {"gate": read(root / "gate.json")["gate"], "by_type_alpha": by_type,
            "noise_null": {"k": k, "n": n, "rate": k / n if n else None, "ci": list(share_ci(k, n))}}


def structural(record):
    p = record["p_struct_corr"]
    return p is not None and p <= 0.01


def report(out):
    shapes = {p.name: shape_report(p) for p in sorted((out / "planted").iterdir()) if p.is_dir()}
    root = out / "pooling"
    pooling = None
    if root.exists():
        trained = records(root / "scores" / "trained.jsonl")
        pools = ("trained", "trained,shuf1", "trained,shuf1,shuf2,shuf3")
        summaries = {pool: read(root / f"summary_{pool}.json") for pool in pools}
        pooling = {"records_identical": trained == records(root / "scores_4" / "trained.jsonl"),
                   "n_trained": len(trained),
                   "trained_calls": {pool: s["pooled_v12"]["calls"]["trained"] for pool, s in summaries.items()},
                   "trained_share_struct": {pool: s["sources"]["trained"]["share_struct"] for pool, s in summaries.items()}}
    result = {"shapes": shapes, "pooling": pooling,
              "inputs_sha256": {name: v["sha256"] for name, v in read(out / "inputs.json").items()}}
    (out / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Ruler checks", "", "## Planted columns", ""]
    for name, shape in shapes.items():
        lines += [f"### {name}", "", "| Gate | Pass/total |", "| --- | --- |"]
        lines += [f"| {key} | {value} |" for key, value in shape["gate"].items()]
        null = shape["noise_null"]
        lines += ["", f"Noise null: {null['k']}/{null['n']}; rate {null['rate']}; 95% Clopper-Pearson {null['ci']}.",
                  "", "| Type@alpha | Calls | Structural k/n (share) | Median primary JSD | Residue modes |",
                  "| --- | --- | --- | --- | --- |"]
        for key, row in shape["by_type_alpha"].items():
            lines.append(f"| {key} | {json.dumps(row['calls'])} | {row['k_struct']}/{row['n']} "
                         f"({row['k_struct'] / row['n']:.6g}) | {row['jsd_q50']} | {json.dumps(row['residue_modes'])} |")
        lines.append("")
    if pooling is not None:
        shares = list(pooling["trained_share_struct"].values())
        lines += ["## Pooling", "", f"Records identical: {pooling['records_identical']}; trained columns: {len(trained)}.",
                  f"Trained share_struct unchanged: {all(s == shares[0] for s in shares)}.", "",
                  "| Pool | Trained calls | Trained share_struct |", "| --- | --- | --- |"]
        lines += [f"| {pool} | {json.dumps(pooling['trained_calls'][pool])} | {pooling['trained_share_struct'][pool]} |"
                  for pool in pools]
    (out / "checks.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    report(ap.parse_args().out)


if __name__ == "__main__":
    main()
