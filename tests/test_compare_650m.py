"""Blind acceptance tests for specs/compare_650m.md, including amendment 1.

Use only the frozen CLI for h3_report so collection needs no implementation.
COMPARE650_SMOKE_BUDGET bounds the smoke job and sequential references together.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
CONTROLS = ("S1", "S2", "S3")
SOURCES = ("N", "R", *CONTROLS, "RS1", "RS2", "RS3")
ATLAS_SOURCES = ("neurons_post", "randdir_512_0")
REPORT_KEYS = {"n", "calibration", "calibration_passes", "d_NR", "d_SRS", "d_NS_mean",
               "h1", "h1_conditions", "secondary", "bootstrap", "repro"}
EXACT = ("top_val", "top_tok", "res_tok", "res_val", "res_key", "sub_bin",
         "enr", "ann_tot", "edges")
SUMS = ("sub_sum", "sub_sq", "gram")


def _env(**extra):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1",
               PYTHONDONTWRITEBYTECODE="1")
    env.update(extra)
    return env


def _run(args, *, env=None, timeout=60):
    result = subprocess.run(list(map(str, args)), cwd=ROOT, env=env or _env(),
                            capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, (args, result.returncode,
                                    result.stdout[-6000:], result.stderr[-6000:])


def _json(path):
    return json.loads(path.read_text())


def _comparison(d, ci):
    return {"d": d, "p": 0.005, "ci": list(ci)}


def _compare_input(nr, calibration, h1):
    """Construct compare_report's JSON schema without invoking that report."""
    return {
        "n": dict.fromkeys(SOURCES, 20),
        "calibration": dict(zip(CONTROLS, calibration)),
        "calibration_passes": True,
        "d_NR": nr,
        "d_SRS": {name: _comparison(0.0, [-0.1, 0.1]) for name in CONTROLS},
        "d_NS_mean": sum(row["d"] for row in calibration) / 3,
        "h1": h1,
        "h1_conditions": {"d_ge_0147": False, "p_lt_001": True,
                          "attributable": False, "half_gap": False},
        "secondary": {
            "share_residue_mode": {name: {"k": 0, "n": 20, "share": 0.0,
                                          "ci": [0.0, 0.16]} for name in SOURCES},
            "median_primary_jsd": dict.fromkeys(SOURCES, 0.1),
            "peak_log2_median": dict.fromkeys(SOURCES, 1.0),
            "pooled_v12_calls": {name: {"no association found by this procedure": 20}
                                for name in SOURCES},
        },
        "bootstrap": {"draws": 2000, "seed": 0, "level": 0.95},
        "repro": None,
    }


def _check_h3(result, small, large, nr_trend, calibration_trends):
    assert set(result) == {"d_NR", "calibration", "h1"}
    assert set(result["calibration"]) == set(CONTROLS)
    assert result["h1"] == {"small": small["h1"], "large": large["h1"]}
    pairs = [(result["d_NR"], small["d_NR"], large["d_NR"], nr_trend)]
    pairs += [(result["calibration"][name], small["calibration"][name],
               large["calibration"][name], calibration_trends[name]) for name in CONTROLS]
    for row, first, second, trend in pairs:
        assert set(row) == {"small", "large", "trend"}
        assert row["small"] == {key: first[key] for key in ("d", "ci")}
        assert row["large"] == {key: second[key] for key in ("d", "ci")}
        assert row["trend"] == trend


@pytest.mark.parametrize("small,large,trend", [
    (_comparison(-0.021, [-0.062, 0.023]), _comparison(0.10, [0.08, 0.12]), "grows"),
    (_comparison(-0.021, [-0.062, 0.023]), _comparison(0.04, [0.02, 0.06]), "flat"),
    (_comparison(0.10, [0.08, 0.12]), _comparison(-0.021, [-0.062, 0.023]), "shrinks"),
    (_comparison(0.04, [0.02, 0.06]), _comparison(-0.021, [-0.062, 0.023]), "flat"),
    (_comparison(0.0, [-0.1, 0.1]), _comparison(0.2, [0.1, 0.3]), "flat"),
    (_comparison(0.2, [0.1, 0.3]), _comparison(0.0, [-0.1, 0.1]), "flat"),
    (_comparison(0.0, [-0.2, 0.2]), _comparison(0.1, [0.0, 0.15]), "flat"),
    (_comparison(0.0, [-0.1, 0.1]), _comparison(0.0, [-0.1, 0.1]), "flat"),
], ids=["worked-grows", "worked-overlap", "mirrored-shrinks", "mirrored-overlap",
        "touch-above", "touch-below", "contained", "identical"])
@pytest.mark.parametrize("rotation", range(3))
def test_h3_hand_built_compare_json(tmp_path, small, large, trend, rotation):
    # Rotate different trends through S1..S3 to detect pooling or aliasing them.
    first = [_comparison(-0.2, [-0.3, -0.1]), _comparison(0.5, [0.4, 0.6]),
             _comparison(0.0, [-0.1, 0.1])]
    second = [_comparison(0.2, [0.1, 0.3]), _comparison(0.1, [0.0, 0.2]),
              _comparison(0.2, [0.1, 0.3])]
    trends = ["grows", "shrinks", "flat"]
    first = first[rotation:] + first[:rotation]
    second = second[rotation:] + second[:rotation]
    trends = trends[rotation:] + trends[:rotation]
    h1_pairs = [("supports", "against"), (None, "weak support"), ("strongly against", None)]
    small_input = _compare_input(small, first, h1_pairs[rotation][0])
    large_input = _compare_input(large, second, h1_pairs[rotation][1])
    small_path, large_path, out = (tmp_path / name for name in
                                   ("small.json", "large.json", "h3.json"))
    small_path.write_text(json.dumps(small_input))
    large_path.write_text(json.dumps(large_input))
    _run([sys.executable, ROOT / "scripts/h3_report.py", "--small", small_path,
          "--large", large_path, "--out", out])
    _check_h3(_json(out), small_input, large_input, trend, dict(zip(CONTROLS, trends)))


def _reference_args(recorded, pilot, out):
    # run.json stores vars(argparse.Namespace), not a shell command. Source caps
    # are append options; the other lists are nargs options (including edge-q).
    args = []
    for name, value in recorded.items():
        if name in ("shard", "out", "pilot_from") or value is None or value is False:
            continue
        flag = "--" + name.replace("_", "-")
        if value is True:
            args.append(flag)
        elif isinstance(value, list):
            if name == "source_cap":
                for entry in value:
                    args.extend([flag, str(entry)])
            elif value:
                args.extend([flag, *map(str, value)])
        else:
            args.extend([flag, str(value)])
    return [*args, "--pilot-from", str(pilot), "--out", str(out)]


def _check_atlas(merged_path, reference_path):
    import torch

    merged = torch.load(merged_path, map_location="cpu", weights_only=False)
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    for key in EXACT:
        assert (key in merged) == (key in reference), key
        if key not in reference:
            # Annotation fields are conditional in Atlas.result().
            assert key in ("enr", "ann_tot"), key
            continue
        x, y = merged[key], reference[key]
        assert x.dtype == y.dtype and x.shape == y.shape, key
        # Compare storage bytes, including NaN payloads and signed zeros.
        assert x.contiguous().numpy().tobytes() == y.contiguous().numpy().tobytes(), key
    for key in SUMS:
        assert (key in merged) == (key in reference), key
        if key not in reference:
            assert key == "gram", key
            continue
        x, y = merged[key], reference[key]
        assert x.dtype == y.dtype and x.shape == y.shape, key
        x, y = x.double(), y.double()
        # Same max-relative-error convention as compare_atlas.py.
        relative = float((x - y).abs().max() / y.abs().max().clamp(min=1e-300)) if x.numel() else 0.0
        assert relative <= 1e-9, (key, relative)
    for key in ("n_seen", "n_sub", "gram_rows", "n_batches", "n_total_plan"):
        assert (key in merged) == (key in reference), key
        if key in reference:
            assert merged[key] == reference[key], key
        else:
            assert key == "gram_rows", key
    assert merged["n_seen"] > 0
    assert merged["cfg"] == reference["cfg"]
    assert "gram" not in merged  # Amendment 1 keeps --no-gram in SMOKE.


def test_compare_650m_smoke(tmp_path):
    budget = int(os.environ.get("COMPARE650_SMOKE_BUDGET", "3600"))
    started = time.monotonic()

    def remaining():
        seconds = budget - (time.monotonic() - started)
        assert seconds > 0, "COMPARE650_SMOKE_BUDGET exhausted"
        return seconds

    data = ROOT / "data/swissprot_test500.pt"
    assert data.is_file(), data
    out, scratch, fs, job = (tmp_path / name for name in ("out", "scratch", "fs", "job"))
    for path in (out, scratch, fs / "cache", job, tmp_path / "tmp", tmp_path / "cache"):
        path.mkdir(parents=True)
    (fs / "cache/swissprot_2026_03.pt").symlink_to(data)
    # Preserve the offline model cache while redirecting generated caches.
    hf_home = os.environ.get("HF_HOME", str(Path(os.environ.get(
        "XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface"))
    env = _env(SMOKE="1", JOB=str(job), OUT=str(out), SCRATCH=str(scratch), FS=str(fs),
               WORKERS="1", PER_LAYER="4", PYTHON=sys.executable, SMOKE_DATA=str(data),
               DEADLINE=str(int(time.time()) + budget - 10), HF_HOME=hf_home,
               TMPDIR=str(tmp_path / "tmp"), XDG_CACHE_HOME=str(tmp_path / "cache"),
               MPLCONFIGDIR=str(tmp_path / "cache/mpl"),
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    env.pop("SMALL_COMPARE", None)  # Exercise amendment 1's self-comparison default.
    _run(["bash", ROOT / "jobs/compare_650m/run.sh"], env=env, timeout=remaining())
    comparison = _json(out / "compare.json")
    assert set(comparison) == REPORT_KEYS
    assert set(comparison["n"]) == set(SOURCES)
    assert set(comparison["calibration"]) == set(CONTROLS)
    assert set(comparison["d_SRS"]) == set(CONTROLS)
    assert set(comparison["h1_conditions"]) == {"d_ge_0147", "p_lt_001", "attributable", "half_gap"}
    assert set(comparison["bootstrap"]) == {"draws", "seed", "level"}
    assert set(comparison["secondary"]) == {
        "share_residue_mode", "median_primary_jsd", "peak_log2_median", "pooled_v12_calls"}
    for values in comparison["secondary"].values():
        assert set(values) == set(SOURCES)
    for row in [comparison["d_NR"], *comparison["calibration"].values(),
                *comparison["d_SRS"].values()]:
        assert set(row) == {"d", "p", "ci"}
    assert comparison["repro"] is None
    _check_h3(_json(out / "h3.json"), comparison, comparison, "flat",
              dict.fromkeys(CONTROLS, "flat"))

    for run in ("trained", "shuf1", "shuf2", "shuf3"):
        run_scratch, run_out = scratch / run, out / run
        pilot = run_scratch / "pilot.pt"
        assert pilot.is_file(), pilot
        assert (run_out / "pilot.json").is_file()
        for index in range(2):
            shard_dir = run_scratch / f"s{index}"
            record = _json(shard_dir / "run.json")
            assert record["complete"] is True
            assert record["args"]["shard"] == [index, 2]
            assert _json(run_out / f"s{index}.run.json") == record
            assert (run_out / f"s{index}.batches.jsonl").read_bytes() == (
                shard_dir / "batches.jsonl").read_bytes()
            for source in ATLAS_SOURCES:
                assert (shard_dir / f"shard_{source}.pt").is_file()
        recorded = _json(run_scratch / "s0/run.json")["args"]
        # One unsharded invocation produces both sources. Never overlap references.
        reference = tmp_path / f"reference_{run}"
        _run([sys.executable, ROOT / "scripts/run_atlas.py",
              *_reference_args(recorded, pilot, reference)], env=env, timeout=remaining())
        assert _json(reference / "run.json")["complete"] is True
        for source in ATLAS_SOURCES:
            assert (run_out / f"merge_{source}.json").is_file()
            _check_atlas(run_scratch / f"atlas_{source}.pt", reference / f"atlas_{source}.pt")
    remaining()
