"""Summarise reproduction and the frozen 8M calibration rule."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def report(out):
    names = ("trained", "shuf1", "shuf2", "shuf3")
    repro = {}
    for name in names:
        row = json.loads((out / f"repro_{name}.json").read_text())
        repro[name] = {key: row[key] for key in ("ok", "exact_all")}
    gate = json.loads((out / "calibration.json").read_text())
    calibration = {}
    for control in names[1:]:
        row = gate["stats"]["primary_jsd"]["vs"][control]
        d, p = row["d"], row["p"]
        # The smoke Kr=8 cannot meet the ruler's minimum of ten clusters.
        passes = d is not None and p is not None and d >= 0.147 and p < 0.01
        calibration[control] = {"d": d, "p": p, "passes": passes}
    trained = json.loads((out / "trained" / "run.json").read_text())
    randdir = trained["sources"]["randdir_512_0"]
    (out / "randdir").mkdir(exist_ok=True)
    (out / "randdir" / "run.json").write_text(json.dumps(randdir, indent=2) + "\n")
    result = {"repro": repro, "calibration": calibration,
              "calibration_passes": all(row["passes"] for row in calibration.values()),
              "randdir": {key: randdir[key] for key in ("digests", "directions_sha256")}}
    (out / "calib.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    report(ap.parse_args().out)


if __name__ == "__main__":
    main()
