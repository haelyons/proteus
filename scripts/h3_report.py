"""Compare frozen small- and large-model effect intervals for H3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def comparison(small, large):
    small = {key: small[key] for key in ("d", "ci")}
    large = {key: large[key] for key in ("d", "ci")}
    trend = "flat"
    if small["ci"] is not None and large["ci"] is not None:
        if large["ci"][0] > small["ci"][1]:
            trend = "grows"
        elif large["ci"][1] < small["ci"][0]:
            trend = "shrinks"
    return {"small": small, "large": large, "trend": trend}


def report(small, large):
    return {"d_NR": comparison(small["d_NR"], large["d_NR"]),
            "calibration": {name: comparison(small["calibration"][name], large["calibration"][name])
                            for name in ("S1", "S2", "S3")},
            "h1": {"small": small["h1"], "large": large["h1"]}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--small", type=Path, required=True)
    ap.add_argument("--large", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = report(json.loads(args.small.read_text()), json.loads(args.large.read_text()))
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
