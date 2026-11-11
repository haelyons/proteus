"""Blind acceptance checks from frozen specs/calib.md.

Run with pytest or ``python -m tests.test_calib`` (exit status 0/1).
Only CLI artifacts are inspected. Historical code is executed, not imported.
Subprocess budgets total 18 minutes; all temporary files stay in the worktree.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch

from tests.test_planted import _input

ROOT = Path(__file__).absolute().parents[1]
SPEC_COMMIT = "cbb273a5fd2ab25ce207b274072a4b3d30fb560b"
CONTROLS = ("shuf1", "shuf2", "shuf3")
RUNS = ("trained", *CONTROLS)


def _env(**extra):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    # Historical subprocesses must import the historical atlas package.
    env.pop("PYTHONPATH", None)
    env.pop("PROTEUS_TEST_SOURCES", None)
    env.update(extra)
    return env


def _run(args, *, cwd=ROOT, env=None, timeout=60):
    result = subprocess.run(list(map(str, args)), cwd=cwd, env=env or _env(),
                            capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, (args, result.returncode,
                                    result.stdout[-6000:], result.stderr[-6000:])
    return result


def _json(path):
    return json.loads(path.read_text())


@pytest.fixture
def work():
    with tempfile.TemporaryDirectory(prefix=".calib-tests-", dir=ROOT) as name:
        yield Path(name)


def _historical(work):
    tree = work / "spec-commit"
    files = _run(["git", "ls-tree", "-r", "--name-only", SPEC_COMMIT,
                  "atlas", "scripts/synthetic_columns.py"]).stdout.splitlines()
    assert "atlas/__init__.py" in files and "scripts/synthetic_columns.py" in files
    for name in files:
        content = subprocess.run(["git", "show", f"{SPEC_COMMIT}:{name}"], cwd=ROOT,
                                 env=_env(), capture_output=True, check=True).stdout
        destination = tree / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    return tree


def _tensors(value, path=()):
    if isinstance(value, torch.Tensor):
        return {path: value}
    found = {}
    items = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(
        value, (list, tuple)) else ()
    for key, child in items:
        found.update(_tensors(child, (*path, key)))
    return found


def _assert_equal(actual, expected, path=()):
    assert type(actual) is type(expected), path
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        assert torch.equal(actual, expected), path
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _assert_equal(actual[key], expected[key], (*path, key))
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (value, reference) in enumerate(zip(actual, expected)):
            _assert_equal(value, reference, (*path, index))
    else:
        assert actual == expected, path


@pytest.mark.parametrize("shape,gen_workers,noise_columns", [
    ("randdir", 1, 3),
    ("randdir", 3, 10),
    ("gauss", 3, 10),
])
def test_randdir_matches_spec_commit(work, record_property, shape, gen_workers, noise_columns):
    data = _input(work / "inputs", n=4, length=64)["path"]
    historical = _historical(work)
    outputs = []
    for name, tree in (("before", historical), ("after", ROOT)):
        out = work / name
        extra = ["--gen-workers", gen_workers] if name == "after" and gen_workers > 1 else []
        _run([sys.executable, "scripts/synthetic_columns.py",
              "--data", data / "swissprot.pt", "--desc-dir", data / "desc",
              "--uniref", data / "uniref.tsv.gz", "--out", out,
              "--shape", shape, "--replicates", 1, "--noise-columns", noise_columns,
              "--tokens-per-batch", 4096, "--pilot-tokens", 256,
              "--Kr", 8, "--seed", 19, "--noise-seed", 7, *extra], cwd=tree)
        outputs.append(torch.load(out / "synthetic_atlas.pt",
                                  map_location="cpu", weights_only=False))
    before, after = outputs
    if gen_workers > 1:
        _assert_equal(after, before)
        record_property(f"{shape}_equality", "bitwise")
        return
    before, after = _tensors(before), _tensors(after)
    assert before and before.keys() == after.keys()
    bitwise, max_abs, max_rel = True, 0.0, 0.0
    for path, expected in before.items():
        actual = after[path]
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        same = torch.equal(actual.contiguous().reshape(-1).view(torch.uint8),
                           expected.contiguous().reshape(-1).view(torch.uint8))
        bitwise = bitwise and same
        if not same:
            assert expected.is_floating_point(), path
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=0, equal_nan=True,
                                       msg=str(path))
            finite = torch.isfinite(expected)
            delta = (actual[finite].double() - expected[finite].double()).abs()
            if delta.numel():
                denominator = expected[finite].double().abs()
                relative = torch.where(denominator == 0, torch.zeros_like(delta), delta / denominator)
                max_abs = max(max_abs, float(delta.max()))
                max_rel = max(max_rel, float(relative.max()))
    record_property("randdir_equality", "bitwise" if bitwise else "relative<=1e-6")
    record_property("randdir_max_abs_diff", max_abs)
    record_property("randdir_max_rel_diff", max_rel)


def _smoke(work, job, timeout, **extra):
    out, scratch, fs = work / "out", work / "scratch", work / "fs"
    data = ROOT / "data/swissprot_test500.pt"
    assert data.is_file(), data
    for path in (out, scratch, fs / "cache", work / "tmp", work / "cache"):
        path.mkdir(parents=True)
    (fs / "cache/swissprot_2026_03.pt").symlink_to(data)
    # Resolve the normal HF cache before redirecting other writable caches.
    # Do not replace HOME or HF_HUB_CACHE: cached model weights are read-only input.
    hf_home = os.environ.get("HF_HOME", str(Path(os.environ.get(
        "XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface"))
    env = _env(SMOKE="1", JOB=f"test-{job}", OUT=str(out), SCRATCH=str(scratch),
               FS=str(fs), DEADLINE=str(int(time.time()) + timeout - 10),
               WORKERS="1", PER_LAYER="4", PYTHON=sys.executable,
               SMOKE_DATA=str(data), PROTEUS_TEST_DATA=str(data),
               TMPDIR=str(work / "tmp"), XDG_CACHE_HOME=str(work / "cache"),
               HF_HOME=hf_home, MPLCONFIGDIR=str(work / "cache" / "mpl"),
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
               **extra)
    _run(["bash", f"jobs/{job}/run.sh"], env=env, timeout=timeout)
    return out


def test_pooling_zero_skips_every_pooling_step(work):
    out = _smoke(work, "ruler_checks", 240, POOLING="0", SHAPES="randdir sae:0.001")
    checks = _json(out / "checks.json")
    assert checks["pooling"] is None
    assert checks["shapes"], "job must still run planted checks"
    steps = _json(out / "timeline.json")["steps"]
    assert steps
    for row in steps:
        assert isinstance(row["step"], str)
        assert not row["step"].startswith("pooling_"), row


def _number(value):
    assert type(value) in (int, float) and math.isfinite(value)


def _sha256(value):
    assert isinstance(value, str) and len(value) == 64
    assert all(c in "0123456789abcdef" for c in value)


def test_calib_smoke_reproduction_and_gate_arithmetic(work):
    out = _smoke(work, "calib_8m", 480)
    report = _json(out / "calib.json")
    assert {"repro", "calibration", "calibration_passes", "randdir"} <= report.keys()
    assert set(report["repro"]) == set(RUNS)
    for name in RUNS:
        repro = report["repro"][name]
        raw = _json(out / f"repro_{name}.json")
        for key in ("ok", "exact_all"):
            assert repro[key] is True, (name, repro)
            assert raw[key] is True, (name, raw)

    calibration = _json(out / "calibration.json")["stats"]["primary_jsd"]["vs"]
    assert set(report["calibration"]) == set(CONTROLS)
    passes = []
    for control in CONTROLS:
        measured, copied = calibration[control], report["calibration"][control]
        assert {"d", "p", "passes"} <= copied.keys()
        d, p = measured["d"], measured["p"]
        _number(d)
        _number(p)
        assert -1 <= d <= 1 and 0 <= p <= 1
        assert copied["d"] == d and copied["p"] == p
        # Calibration requires trained > control, not the two-sided separation gate.
        passes.append(d >= 0.147 and p < 0.01)
        assert copied["passes"] is passes[-1]
    assert report["calibration_passes"] is all(passes)

    randdir = report["randdir"]
    assert {"digests", "directions_sha256"} <= randdir.keys()
    assert isinstance(randdir["digests"], dict) and randdir["digests"]
    for digest in randdir["digests"].values():
        _sha256(digest)
    _sha256(randdir["directions_sha256"])


def main():
    return 0 if pytest.main([str(Path(__file__)), "-q", *sys.argv[1:]]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
