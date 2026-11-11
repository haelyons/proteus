"""Evaluate source separation and planted-mixture checks from a ruler_checks directory."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.compare import share_ci  # noqa: E402

SOURCES = ("trained", "shuf1", "shuf2", "shuf3")
STATS = ("share_struct", "share_residue_mode", "primary_jsd")


def _rank_counts(x, y):
    x, y = list(x), list(y)
    if not x or not y:
        raise ValueError("samples must be non-empty")
    pooled = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    u, ties, below = 0.0, 0, 0
    i = 0
    while i < len(pooled):
        j = i + 1
        while j < len(pooled) and pooled[j][0] == pooled[i][0]:
            j += 1
        nx = sum(source == 0 for _, source in pooled[i:j])
        ny = j - i - nx
        u += nx * (below + ny / 2)
        below += ny
        ties += (j - i) ** 3 - (j - i)
        i = j
    return u, ties, len(x), len(y)


def mann_whitney(x, y):
    """Two-sided, tie-corrected normal approximation without continuity correction."""
    u, ties, n1, n2 = _rank_counts(x, y)
    n = n1 + n2
    variance = n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1)))
    z = (u - n1 * n2 / 2) / math.sqrt(variance) if variance > 0 else 0.0
    return {"U": u, "z": z, "p": math.erfc(abs(z) / math.sqrt(2))}


def cliffs_delta(x, y):
    """Fraction of greater pairs minus fraction of lesser pairs."""
    u, _, n1, n2 = _rank_counts(x, y)
    return 2 * u / (n1 * n2) - 1


def ci_separation(a, b):
    """Strict separation of two intervals; touching intervals do not separate."""
    return 1 if a[0] > b[1] else -1 if a[1] < b[0] else 0


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stat_summary(rows, stat):
    scored = [r for r in rows if not r["insufficient"]]
    n = len(scored)
    if stat == "primary_jsd":
        values = [r[stat] for r in scored]
        return {"n": n, "median": statistics.median(values) if n else None}
    k = sum(r["p_struct_corr"] <= 0.01 if stat == "share_struct" else bool(r["residue_modes"])
            for r in scored)
    return {"k": k, "n": n, "share": k / n if n else None, "ci": list(share_ci(k, n))}


def mixture_report(root):
    columns = {c["col"]: c for c in json.loads((root / "columns.json").read_text())}
    rows = records(root / "scores" / f"{root.name}.jsonl")
    groups = defaultdict(list)
    for r in rows:
        c = columns[r["unit"]]
        if c["alpha"] >= 2:
            groups[c["type"]].append(r)
    parts = {c["type"]: c["parts"] for c in columns.values()
             if c["type"].startswith(("mix_", "part_"))}
    result = {}
    for kind, members in sorted(parts.items()):
        if not kind.startswith("mix_"):
            continue
        # Each part type stores a singleton list of [residue, state] pairs.
        part_types = [[name for name, entries in parts.items()
                       if name.startswith("part_") and entries == [member]] for member in members]
        result[kind] = {}
        for stat in STATS:
            field = "median" if stat == "primary_jsd" else "share"
            mixture = stat_summary(groups[kind], stat)[field]
            values = [stat_summary(groups[name], stat)[field] for names in part_types for name in names]
            complete = bool(part_types) and all(part_types) and all(v is not None for v in values)
            parts_max = max(values) if complete else None
            result[kind][stat] = {"mixture": mixture, "parts_max": parts_max,
                                  "above": mixture is not None and parts_max is not None and mixture > parts_max}
    return result


def report(checks):
    checks = Path(checks)
    rows, peaks = {}, {}
    for source in SOURCES:
        root = checks / "pooling" / "scores"
        rows[source] = records(root / f"{source}.jsonl")
        scores = torch.load(root / f"{source}.scores.pt", map_location="cpu", weights_only=False)
        if len(scores) != len(rows[source]):
            raise ValueError(f"score and record counts differ: {source}")
        values = [max(s["peak"][r["primary"]]["log2"]) for r, s in zip(rows[source], scores)
                  if not r["insufficient"]]
        peaks[source] = statistics.median(values) if values else None
    mixtures = {p.name: mixture_report(p) for p in sorted((checks / "planted").iterdir()) if p.is_dir()}
    stats, qualifies = {}, {}
    for stat in STATS:
        by_source = {source: stat_summary(rows[source], stat) for source in SOURCES}
        vs = {}
        for control in SOURCES[1:]:
            if stat == "primary_jsd":
                x, y = ([r[stat] for r in rows[source] if not r["insufficient"]]
                        for source in ("trained", control))
                if x and y:
                    comparison = {**mann_whitney(x, y), "d": cliffs_delta(x, y)}
                    d = comparison["d"]
                    sign = (1 if d > 0 else -1) if comparison["p"] < 0.01 and abs(d) >= 0.147 else 0
                else:
                    comparison = dict.fromkeys(("U", "z", "p", "d"))
                    sign = 0
            else:
                trained, other = by_source["trained"]["ci"], by_source[control]["ci"]
                comparison = {"lo_hi_trained": trained, "lo_hi_control": other}
                sign = ci_separation(trained, other)
            vs[control] = {**comparison, "sign": sign}
        signs = [v["sign"] for v in vs.values()]
        separates = signs[0] != 0 and len(set(signs)) == 1
        above = any(mix[stat]["above"] for shape in mixtures.values() for mix in shape.values())
        stats[stat] = {"by_source": by_source, "vs": vs, "separates": separates,
                       "direction": signs[0] if separates else 0, "rates_mixtures_above_parts": above}
        qualifies[stat] = separates and not above
    return {"stats": stats, "mixtures": mixtures, "qualifies": qualifies, "peak_log2_median": peaks}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.write_text(json.dumps(report(args.checks), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
