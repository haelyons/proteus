"""Blind acceptance checks from frozen specs/compare_8m.md.

Implementation imports are deferred so collection works before compare_report lands.
The smoke job builds seven small CPU atlases; COMPARE_SMOKE_BUDGET defaults to 3600s.
All generated files stay in the worktree.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median

import pytest
import torch

ROOT = Path(__file__).absolute().parents[1]
CONTROLS = ("S1", "S2", "S3")
SOURCES = ("N", "R", *CONTROLS, "RS1", "RS2", "RS3")
REPORT_KEYS = {"n", "calibration", "calibration_passes", "d_NR", "d_SRS", "d_NS_mean",
               "h1", "h1_conditions", "secondary", "bootstrap", "repro"}


@pytest.fixture
def report_module():
    spec = importlib.util.spec_from_file_location("compare_report", ROOT / "scripts/compare_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def work():
    with tempfile.TemporaryDirectory(prefix=".compare-8m-tests-", dir=ROOT) as name:
        yield Path(name)


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
    return result


def _json(path):
    return json.loads(path.read_text())


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("d,p,ci,ns,srs,expected", [
    (0.30, 1e-10, [0.26, 0.34], [0.236, 0.214, 0.198],
     [[-0.05, 0.03], [-0.02, 0.06], [0.00, 0.08]], "supports"),
    (0.30, 1e-10, [0.26, 0.34], [0.236, 0.214, 0.198],
     [[-0.05, 0.03], [-0.02, 0.06], [0.20, 0.26]], "gap not attributable to training"),
    (0.15, 1e-10, [0.11, 0.19], [0.3, 0.4, 0.5], [[-0.05, 0.09]] * 3, "weak support"),
    (0.10, 1e-10, [0.06, 0.14], [0.3, 0.4, 0.5], [[-0.05, 0.09]] * 3, "against"),
    (-0.20, 1e-5, [-0.24, -0.16], [0.3, 0.4, 0.5], [[-0.05, 0.09]] * 3, "strongly against"),
    (0.147, 0.0099, [0.12, 0.18], [0.1, 0.2, 0.3], [[-0.05, 0.09]] * 3, "supports"),
    (0.147, 0.01, [0.12, 0.18], [0.1, 0.2, 0.3], [[-0.05, 0.09]] * 3, "against"),
    (-0.147, 0.001, [-0.2, -0.1], [0.1, 0.2, 0.3], [[-0.05, 0.09]] * 3, "strongly against"),
    (-0.147, 0.01, [-0.2, -0.1], [0.1, 0.2, 0.3], [[-0.05, 0.09]] * 3, "against"),
    (-0.146999, 0.001, [-0.2, -0.1], [0.1, 0.2, 0.3], [[-0.05, 0.09]] * 3, "against"),
    (0.146999, 0.0099, [0.12, 0.18], [0.1, 0.2, 0.3], [[-0.05, 0.09]] * 3, "against"),
    (0.2, 0.001, [0.11, 0.3], [0.25, 0.5, 0.5], [[-0.05, 0.09]] * 3, "weak support"),
    (0.25, 0.001, [0.11, 0.3], [0.25, 0.5, 0.75], [[-0.05, 0.09]] * 3, "supports"),
])
def test_h1_worked_examples_and_boundaries(report_module, d, p, ci, ns, srs, expected):
    assert report_module.h1_verdict(d, p, ci, ns, srs) == expected


@pytest.mark.parametrize("control", range(3))
def test_h1_requires_strict_separation_from_every_control(report_module, control):
    intervals = [[-0.05, 0.09] for _ in CONTROLS]
    intervals[control] = [0.1, 0.26]
    assert report_module.h1_verdict(0.3, 1e-10, [0.26, 0.34], [0.2] * 3,
                                     intervals) == "gap not attributable to training"


def _bootstrap(x, y, *, draws, seed, level):
    from scripts.separation_gate import cliffs_delta

    generator = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(draws):
        ix = torch.randint(len(x), (len(x),), generator=generator)
        iy = torch.randint(len(y), (len(y),), generator=generator)
        values.append(cliffs_delta([x[i] for i in ix], [y[i] for i in iy]))
    return torch.quantile(torch.tensor(values, dtype=torch.float64),
                          torch.tensor([(1 - level) / 2, (1 + level) / 2],
                                       dtype=torch.float64)).tolist()


@pytest.mark.parametrize("x,y,expected", [
    ([1, 2, 3], [4, 5], [-1.0, -1.0]),
    ([4, 5], [1, 2, 3], [1.0, 1.0]),
    ([2, 2], [2, 2, 2], [0.0, 0.0]),
])
def test_bootstrap_constant_draws(report_module, x, y, expected):
    assert report_module.bootstrap_cliffs(x, y) == expected


def test_bootstrap_defaults(report_module):
    x, y = [0.0, 3.0, 1.0, 1.0], [2.0, 0.0, 4.0]
    assert report_module.bootstrap_cliffs(x, y) == _bootstrap(x, y, draws=2000, seed=0, level=0.95)


@pytest.mark.parametrize("draws,seed,level", [(1, 0, 0.95), (1, 29, 0.8),
                                              (37, 0, 0.95), (61, 29, 0.8)])
def test_bootstrap_draw_order_quantiles_and_repeatability(report_module, draws, seed, level):
    x, y = [3.0, 0.0, 1.0, 1.0, 4.0], [1.0, 5.0, 0.0]
    options = dict(draws=draws, seed=seed, level=level)
    expected = _bootstrap(x, y, **options)
    state = torch.random.get_rng_state().clone()
    actual = report_module.bootstrap_cliffs(x, y, **options)
    assert isinstance(actual, list) and len(actual) == 2
    assert all(type(value) is float for value in actual)
    assert actual == expected
    assert actual == report_module.bootstrap_cliffs(x, y, **options)
    assert -1 <= actual[0] <= actual[1] <= 1
    assert torch.equal(torch.random.get_rng_state(), state)  # uint8 RNG bytes
    assert x == [3.0, 0.0, 1.0, 1.0, 4.0] and y == [1.0, 5.0, 0.0]
    if draws == 1:
        assert actual[0] == actual[1]


def _synthetic(work, calibration, reverse):
    from atlas.compare import summarise

    root = work / "scores"
    root.mkdir()
    records, scores = {}, {}
    for index, name in enumerate(SOURCES):
        n = 20 if name == "N" else 17 + index
        base = 0.6 if name == "N" else 0.1 + index * 0.015
        if name == "R" and reverse:
            base = 0.9
        if name in CONTROLS and (calibration == "none" or
                                 calibration == "one_fails" and name == "S3"):
            base = 0.9
        rows, entries = [], []
        for i in range(n):
            primary = "state5" if i % 2 else "burial3"
            peak = float((1, 3, 7, 100)[i % 4] + index)
            score = {"peak": {primary: {"log2": [-99.0, peak, -1.0]},
                              "rama4": {"log2": [999.0]}}}
            row = dict(layer=i % 2, col=n - i, insufficient=False, primary=primary,
                       primary_jsd=base + 0.01 * ((i * 3) % 7),
                       residue_modes=["K", "E"] if i % 3 == 0 else [],
                       p_struct_corr=0.01 if i % 4 == 0 else 0.5)
            rows.append(row)
            entries.append(score)
        # Interleave a poison observation: filter before reading primary or peak.
        rows.insert(2, dict(layer=0, col=0, insufficient=True, primary=None,
                            primary_jsd=999.0, residue_modes=["K"], p_struct_corr=0.0))
        entries.insert(2, {"peak": {}})
        for row, score in zip(rows, entries):
            row["score_sha256"] = hashlib.sha256(json.dumps(score, sort_keys=True).encode()).hexdigest()
        (root / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        torch.save(entries, root / f"{name}.scores.pt")
        records[name], scores[name] = rows, entries
    # Match compare_sources.summary's format without requiring full ruler scores.
    counts = {name: {"no association found by this procedure": len(rows) - 1,
                     "insufficient samples": 1} for name, rows in records.items()}
    summary = {"sources": {name: summarise(rows) for name, rows in records.items()},
               "pooled_v12": {"pool": list(SOURCES), "n_columns": sum(map(len, records.values())),
                              "calls": counts},
               "tables_sha256": "a" * 64, "ruler_sha256": "b" * 64, "B": 2000}
    path = work / "summary.json"
    path.write_text(json.dumps(summary))
    return root, path, records, scores, counts


def _check_verdict(report, module):
    assert set(report) == REPORT_KEYS
    assert set(report["n"]) == set(SOURCES)
    assert set(report["calibration"]) == set(CONTROLS)
    assert set(report["d_SRS"]) == set(CONTROLS)
    calibration = [report["calibration"][name] for name in CONTROLS]
    passes = all(row["d"] >= 0.147 and row["p"] < 0.01 for row in calibration)
    assert report["calibration_passes"] is passes
    ns = [row["d"] for row in calibration]
    assert report["d_NS_mean"] == pytest.approx(mean(ns), abs=1e-15)
    nr = report["d_NR"]
    intervals = [report["d_SRS"][name]["ci"] for name in CONTROLS]
    assert report["h1_conditions"] == {
        "d_ge_0147": nr["d"] >= 0.147,
        "p_lt_001": nr["p"] < 0.01,
        "attributable": all(nr["ci"][0] > ci[1] for ci in intervals),
        "half_gap": nr["d"] >= 0.5 * mean(ns),
    }
    expected = module.h1_verdict(nr["d"], nr["p"], nr["ci"], ns, intervals) if passes else None
    assert report["h1"] == expected


def _check_arithmetic(report, records, scores, options):
    from scripts.separation_gate import cliffs_delta, mann_whitney, stat_summary

    values = {name: [row["primary_jsd"] for row in rows if not row["insufficient"]]
              for name, rows in records.items()}
    assert report["n"] == {name: len(value) for name, value in values.items()}
    comparisons = [(report["d_NR"], "N", "R")]
    comparisons += [(report["calibration"][name], "N", name) for name in CONTROLS]
    comparisons += [(report["d_SRS"][name], name, "R" + name) for name in CONTROLS]
    for comparison, first, second in comparisons:
        x, y = values[first], values[second]
        assert comparison["d"] == cliffs_delta(x, y), (first, second)
        assert comparison["p"] == mann_whitney(x, y)["p"], (first, second)
        assert comparison["ci"] == _bootstrap(x, y, **options), (first, second)
    assert report["bootstrap"] == options
    secondary = report["secondary"]
    assert set(secondary) == {"share_residue_mode", "median_primary_jsd", "peak_log2_median",
                              "pooled_v12_calls"}
    for name in SOURCES:
        assert secondary["share_residue_mode"][name] == stat_summary(records[name], "share_residue_mode")
        assert secondary["median_primary_jsd"][name] == median(values[name])
        peaks = [max(score["peak"][row["primary"]]["log2"])
                 for row, score in zip(records[name], scores[name]) if not row["insufficient"]]
        assert secondary["peak_log2_median"][name] == median(peaks)


@pytest.mark.parametrize("calibration", ["passes", "one_fails", "none"])
@pytest.mark.parametrize("reverse", [False, True])
def test_compare_report_hand_built_scores(work, report_module, calibration, reverse):
    root, summary, records, scores, counts = _synthetic(work, calibration, reverse)
    options = dict(draws=43, seed=19, level=0.8)
    destination = work / "compare.json"
    args = [sys.executable, "scripts/compare_report.py", "--scores", root,
            "--summary", summary, "--draws", options["draws"], "--seed", options["seed"],
            "--level", options["level"], "--out", destination]
    repro = {name: {"n": 21, "identical": 20} for name in ("N", *CONTROLS)} if reverse else None
    if repro is not None:
        repro_path = work / "repro.json"
        repro_path.write_text(json.dumps(repro))
        args.extend(["--repro", repro_path])
    _run(args)
    result = _json(destination)
    _check_verdict(result, report_module)
    _check_arithmetic(result, records, scores, options)
    assert result["calibration_passes"] is (calibration == "passes")
    assert result["repro"] == repro
    assert result["secondary"]["pooled_v12_calls"] == counts
    assert destination.with_suffix(".md").read_text().strip()


def test_compare_8m_smoke(work, report_module):
    from atlas.ruler import calls

    budget = int(os.environ.get("COMPARE_SMOKE_BUDGET", "3600"))
    data = ROOT / "data/swissprot_test500.pt"
    assert data.is_file(), data
    out, scratch, fs = work / "out", work / "scratch", work / "fs"
    for path in (out, scratch, fs / "cache", work / "tmp", work / "cache"):
        path.mkdir(parents=True)
    (fs / "cache/swissprot_2026_03.pt").symlink_to(data)
    # Preserve the offline HF cache while keeping generated caches inside this worktree.
    hf_home = os.environ.get("HF_HOME", str(Path(os.environ.get(
        "XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface"))
    env = _env(SMOKE="1", JOB="test-compare-8m", OUT=str(out), SCRATCH=str(scratch),
               FS=str(fs), CALIB=str(out / "smoke_inputs"), WORKERS="1", PER_LAYER="4",
               DEADLINE=str(int(time.time()) + budget - 10), PYTHON=sys.executable,
               SMOKE_DATA=str(data), HF_HOME=hf_home, TMPDIR=str(work / "tmp"),
               XDG_CACHE_HOME=str(work / "cache"), MPLCONFIGDIR=str(work / "cache/mpl"),
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    _run(["bash", "jobs/compare_8m/run.sh"], env=env, timeout=budget)
    result = _json(out / "compare.json")
    _check_verdict(result, report_module)
    assert result["bootstrap"] == dict(draws=2000, seed=0, level=0.95)
    assert (out / "compare.md").read_text().strip()
    repro = _json(out / "repro.json")
    assert result["repro"] == repro
    assert set(repro) == {"N", *CONTROLS}
    sample = _json(out / "sample.json")
    records = {name: _rows(out / "scores" / f"{name}.jsonl") for name in SOURCES}
    for name, reference in (("N", "trained"), *((name, "shuf" + name[1:]) for name in CONTROLS)):
        before = _rows(out / "smoke_inputs/pooling/scores" / f"{reference}.jsonl")
        before = {(row["layer"], row["col"]): row["score_sha256"] for row in before}
        after = {(row["layer"], row["col"]): row["score_sha256"] for row in records[name]}
        assert len(after) == len(records[name]) > 0
        assert after == before, name
        assert repro[name] == {"n": len(after), "identical": len(after)}
    pooled, owners = [], []
    for name in SOURCES:
        entries = torch.load(out / "scores" / f"{name}.scores.pt", map_location="cpu", weights_only=False)
        assert len(entries) == len(records[name])
        pooled.extend(entries)
        owners.extend([name] * len(entries))
    counts = {name: Counter() for name in SOURCES}
    for name, call in zip(owners, calls(pooled, rule="v1.2")):
        counts[name][call["call"]] += 1
    assert result["secondary"]["pooled_v12_calls"] == {name: dict(count) for name, count in counts.items()}
    digests = _json(out / "atlases/digests.json")
    for name in ("RS1", "RS2", "RS3"):
        copied = out / "atlases" / f"{name}.pt"
        original = Path(sample["sources"][name]["atlas"])
        # File bytes also preserve NaN padding; floating tensor equality would not.
        assert copied.read_bytes() == original.read_bytes(), name
        assert hashlib.sha256(copied.read_bytes()).hexdigest() in json.dumps(digests)
