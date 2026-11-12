"""Report the frozen eight-source 8M comparison and H1 verdict."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.separation_gate import cliffs_delta, mann_whitney, records, stat_summary  # noqa: E402

SOURCES = ("N", "R", "S1", "S2", "S3", "RS1", "RS2", "RS3")
CONTROLS = ("S1", "S2", "S3")


def bootstrap_cliffs(x, y, *, draws=2000, seed=0, level=0.95):
    """Percentile interval with independent resampling in the frozen draw order."""
    if not x or not y:
        raise ValueError("samples must be non-empty")
    if draws < 1 or not 0 < level < 1:
        raise ValueError("draws must be positive and level must be between zero and one")
    g = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(draws):
        ix = torch.randint(len(x), (len(x),), generator=g)
        iy = torch.randint(len(y), (len(y),), generator=g)
        values.append(cliffs_delta([x[i] for i in ix], [y[i] for i in iy]))
    return torch.quantile(torch.tensor(values, dtype=torch.float64),
                          torch.tensor([(1 - level) / 2, (1 + level) / 2], dtype=torch.float64)).tolist()


def h1_verdict(d_NR, p_NR, ci_NR, d_NS, ci_SRS):
    """Apply the sealed rule with amendment 1, in order."""
    if d_NR <= -0.147 and p_NR < 0.01:
        return "strongly against"
    if d_NR < 0.147 or p_NR >= 0.01:
        return "against"
    if not all(ci_NR[0] > ci[1] for ci in ci_SRS):
        return "gap not attributable to training"
    return "supports" if d_NR >= 0.5 * statistics.mean(d_NS) else "weak support"


def report(scores_dir, summary, *, repro=None, draws=2000, seed=0, level=0.95):
    rows, values, peaks = {}, {}, {}
    for name in SOURCES:
        rows[name] = records(scores_dir / f"{name}.jsonl")
        values[name] = [r["primary_jsd"] for r in rows[name] if not r["insufficient"]]
        scores = torch.load(scores_dir / f"{name}.scores.pt", map_location="cpu", weights_only=False)
        if len(scores) != len(rows[name]):
            raise ValueError(f"score and record counts differ: {name}")
        peak = [max(s["peak"][r["primary"]]["log2"]) for r, s in zip(rows[name], scores)
                if not r["insufficient"]]
        peaks[name] = statistics.median(peak) if peak else None
    pooled = summary["pooled_v12"]
    if len(pooled["pool"]) != len(SOURCES) or set(pooled["pool"]) != set(SOURCES):
        raise ValueError("summary must pool all eight sources")

    def comparison(first, second):
        x, y = values[first], values[second]
        if not x or not y:
            return dict.fromkeys(("d", "p", "ci"))
        return {"d": cliffs_delta(x, y), "p": mann_whitney(x, y)["p"],
                "ci": bootstrap_cliffs(x, y, draws=draws, seed=seed, level=level)}

    calibration = {name: comparison("N", name) for name in CONTROLS}
    passes = all(c["d"] is not None and c["d"] >= 0.147 and c["p"] < 0.01
                 for c in calibration.values())
    nr = comparison("N", "R")
    srs = {name: comparison(name, "R" + name) for name in CONTROLS}
    ds = [c["d"] for c in calibration.values()]
    mean = statistics.mean(ds) if all(d is not None for d in ds) else None
    conditions = {
        "d_ge_0147": nr["d"] is not None and nr["d"] >= 0.147,
        "p_lt_001": nr["p"] is not None and nr["p"] < 0.01,
        "attributable": nr["ci"] is not None and all(
            c["ci"] is not None and nr["ci"][0] > c["ci"][1] for c in srs.values()),
        "half_gap": nr["d"] is not None and mean is not None and nr["d"] >= 0.5 * mean,
    }
    complete = nr["d"] is not None and all(c["ci"] is not None for c in srs.values())
    h1 = h1_verdict(nr["d"], nr["p"], nr["ci"], ds, [c["ci"] for c in srs.values()]) if passes and complete else None
    return {"n": {name: len(values[name]) for name in SOURCES},
            "calibration": calibration, "calibration_passes": passes, "d_NR": nr,
            "d_SRS": srs, "d_NS_mean": mean, "h1": h1, "h1_conditions": conditions,
            "secondary": {
                "share_residue_mode": {name: stat_summary(rows[name], "share_residue_mode") for name in SOURCES},
                "median_primary_jsd": {name: stat_summary(rows[name], "primary_jsd")["median"] for name in SOURCES},
                "peak_log2_median": peaks, "pooled_v12_calls": pooled["calls"]},
            "bootstrap": {"draws": draws, "seed": seed, "level": level}, "repro": repro}


def markdown(result):
    lines = ["# 8M comparison", "", f"Calibration passes: {result['calibration_passes']}",
             f"H1: {result['h1']}", "", "| Comparison | Cliff's d | p | Bootstrap CI |",
             "| --- | --- | --- | --- |"]
    for name, row in [(f"N vs {s}", result["calibration"][s]) for s in CONTROLS] + [
            ("N vs R", result["d_NR"])] + [(f"{s} vs R{s}", result["d_SRS"][s]) for s in CONTROLS]:
        lines.append(f"| {name} | {row['d']} | {row['p']} | {row['ci']} |")
    lines += ["", f"Mean d(N, Sc): {result['d_NS_mean']}",
              f"H1 conditions: {json.dumps(result['h1_conditions'], sort_keys=True)}", "",
              "| Source | n | Residue modes (k/n; share; CI) | Median primary JSD | Median peak log2 | Pooled v1.2 calls |",
              "| --- | --- | --- | --- | --- | --- |"]
    secondary = result["secondary"]
    for name in SOURCES:
        share = secondary["share_residue_mode"][name]
        lines.append(f"| {name} | {result['n'][name]} | {share['k']}/{share['n']}; {share['share']}; {share['ci']} | "
                     f"{secondary['median_primary_jsd'][name]} | {secondary['peak_log2_median'][name]} | "
                     f"{json.dumps(secondary['pooled_v12_calls'][name], sort_keys=True)} |")
    lines += ["", f"Bootstrap: {json.dumps(result['bootstrap'], sort_keys=True)}",
              f"Reproduction: {json.dumps(result['repro'], sort_keys=True)}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--repro", type=Path)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = report(args.scores, json.loads(args.summary.read_text()),
                    repro=json.loads(args.repro.read_text()) if args.repro else None,
                    draws=args.draws, seed=args.seed, level=args.level)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.out.with_suffix(".md").write_text(markdown(result))


if __name__ == "__main__":
    main()
