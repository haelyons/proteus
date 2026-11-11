"""Blind checks from specs/ruler_checks.md; no models or real input data.

Run with pytest or ``python -m tests.test_ruler_checks`` (exit status 0/1).
Report input filenames are not frozen. Fixtures use planted/SHAPE/{columns.json,
calls.json,gate.json,scores/SHAPE.jsonl,summary.json}, pooling/{scores,scores_4}/
trained.jsonl and pooling/summary_POOL.json. POOL is the comma-separated pool.
Only checks.json's frozen keys are asserted; no implementation helpers imported.
"""
from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import pytest

from tests.test_planted import _input

ROOT = Path(__file__).absolute().parents[1]
POOLS = ("trained", "trained,shuf1", "trained,shuf1,shuf2,shuf3")
NONE = "no association found by this procedure"


def _env(**extra):
    return dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", **extra)


def _run(args, *, env=None, timeout=60):
    result = subprocess.run(list(map(str, args)), cwd=ROOT, env=env or _env(),
                            capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, (args, result.returncode, result.stdout, result.stderr)
    return result


def _json(path):
    return json.loads(path.read_text())


def _write(path, value, *, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in value)
                    if jsonl else json.dumps(value))


@pytest.fixture
def work():
    # Keep even synthetic inputs and temporary files inside this worktree.
    with tempfile.TemporaryDirectory(prefix="ruler-checks-", dir=ROOT) as name:
        yield Path(name)


def test_calls_adapter_matches_run_ruler_every_field(work):
    data = _input(work / "inputs", n=32, length=64)["path"]
    planted, sample, scores = work / "planted", work / "sample.json", work / "scores"
    atlas = planted / "synthetic_atlas.pt"
    _run([sys.executable, "scripts/synthetic_columns.py", "--data", data / "swissprot.pt",
          "--desc-dir", data / "desc", "--uniref", data / "uniref.tsv.gz",
          "--replicates", 1, "--tokens-per-batch", 2048, "--pilot-tokens", 2048,
          "--Kr", 32, "--out", planted])
    columns = _json(planted / "columns.json")
    c = len(columns)
    assert c > 0
    reference, adapted = work / "reference.json", work / "adapted.json"
    _run([sys.executable, "scripts/run_ruler.py", "--atlas", atlas,
          "--data", data / "swissprot.pt", "--desc-dir", data / "desc",
          "--uniref", data / "uniref.tsv.gz", "--rule", "v1.2", "--B", 200,
          "--out", reference], timeout=60)
    cli = [sys.executable, "scripts/compare_sources.py"]
    _run([*cli, "sample", "--atlas", f"synthetic={atlas}", "--per-layer", c,
          "--seed", 0, "--out", sample])
    _run([*cli, "score", "--sample", sample, "--source", "synthetic", "--atlas", atlas,
          "--desc", data / "desc", "--data", data / "swissprot.pt",
          "--uniref", data / "uniref.tsv.gz", "--B", 200, "--workers", 1,
          "--out", scores], timeout=60)
    _run([*cli, "calls", "--scores", scores, "--source", "synthetic", "--rule", "v1.2",
          "--out", adapted])
    expected, actual = _json(reference), _json(adapted)
    assert expected["rule"] == actual["rule"] == "v1.2"
    assert actual["residues"] is None or actual["residues"] == expected["residues"]
    assert type(actual["seconds"]) in (int, float) and actual["seconds"] >= 0
    assert len(actual["calls"]) == len(expected["calls"]) == c
    assert {row["col"] for row in actual["calls"]} == set(range(c))
    sampled = _json(sample)["sources"]["synthetic"]["columns"]
    assert [row["col"] for row in actual["calls"]] == [unit for layer, unit in sampled]
    # Full sampling still permutes columns. Compare complete dictionaries after
    # joining by unit; this includes levels and every key run_ruler writes.
    assert sorted(actual["calls"], key=lambda row: row["col"]) == expected["calls"]


def _cp(k, n):
    """Invert exact binomial tails by bisection, independently of beta/scipy."""
    def root(start, target):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            p = (lo + hi) / 2
            tail = math.fsum(math.comb(n, j) * p ** j * (1 - p) ** (n - j)
                             for j in range(start, n + 1))
            if tail < target:
                lo = p
            else:
                hi = p
        return (lo + hi) / 2

    return (0.0 if k == 0 else root(k, 0.025),
            1.0 if k == n else root(k + 1, 0.975))


def _record(col, p, jsd, residues=(), source="gauss"):
    return dict(source=source, layer=0, unit=col, col=col, seed=col, pole=0,
                insufficient=False, n_clusters_peak=20, primary="state5", n_desc=2,
                p_struct=p / 2, p_struct_corr=p, primary_jsd=jsd, p_res=0.5,
                residue_modes=list(residues), score_sha256=f"{col:064x}")


def _report_inputs(out, changed):
    planted = out / "planted" / "gauss"
    columns, records, calls = [], [], []

    def add(kind, alpha, p, jsd, residues=(), call=NONE):
        col = len(columns)
        columns.append(dict(col=col, type=kind, alpha=alpha, rep=col))
        records.append(_record(col, p, jsd, residues))
        calls.append(dict(col=col, call=call, residue_modes=list(residues)))

    # 18 base noise columns at all six alphas, plus 500 extra alpha=0.
    # Includes exact threshold and just-above-threshold p-values.
    for i in range(518):
        add("noise", (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)[i // 3] if i < 18 else 0.0,
            {0: 0.01, 1: 0.009, 3: 0.002, 17: 0.0}.get(i, 0.010001), 0.02)
    for p, jsd, residues, call in (
            (0.01, 0.02, ["K", "E"], "context association"),
            (0.010001, 0.08, [], NONE),
            (0.001, 0.12, ["K"], "structural association"),
            (0.8, 0.30, ["E"], "composition association")):
        add("mix_same_2", 1.0, p, jsd, residues, call)
    add("mix_same_2", 2.0, 0.02, 0.4, ["R"], NONE)
    add("part_same_0", 1.0, 0.003, 0.07, ["K"], "context association")
    add("part_same_0", 1.0, 0.9, 0.11, [], NONE)
    gate = {"noise + every alpha = 0": "518/518"}
    _write(planted / "columns.json", columns)
    _write(planted / "calls.json", {"rule": "v1.2", "calls": calls,
                                    "residues": 2048, "seconds": 0.0})
    _write(planted / "gate.json", {"gate": gate})
    # Reverse record order to catch positional joins to columns.json.
    _write(planted / "scores" / "gauss.jsonl", records[::-1], jsonl=True)
    _write(planted / "summary.json", {"sources": {"gauss": {}},
                                     "pooled_v12": {"pool": ["gauss"],
                                                    "n_columns": len(records),
                                                    "calls": {"gauss": dict(Counter(
                                                        row["call"] for row in calls))}}})
    trained = [_record(i, p, jsd, source="trained") for i, (p, jsd) in enumerate(
        [(0.01, 0.12), (0.010001, 0.05), (0.002, 0.08)])]
    repeat = copy.deepcopy(trained)
    if changed:
        repeat[1][changed] = {"score_sha256": "f" * 64, "n_clusters_peak": 21,
                              "residue_modes": ["K"]}[changed]
    pooling = out / "pooling"
    _write(pooling / "scores" / "trained.jsonl", trained, jsonl=True)
    _write(pooling / "scores_4" / "trained.jsonl", repeat, jsonl=True)
    counts = [{NONE: 2, "structural association": 1},
              {NONE: 1, "structural association": 2}, {"structural association": 3}]
    for pool, count in zip(POOLS, counts):
        _write(pooling / f"summary_{pool}.json", {
            "sources": {"trained": {"n_sampled": 3, "n_scored": 3, "k_struct": 2,
                                    "share_struct": 2 / 3}},
            "pooled_v12": {"pool": pool.split(","), "n_columns": 3 * len(pool.split(",")),
                           "calls": {"trained": count}}})
    digest = "a" * 64
    _write(out / "inputs.json", {"synthetic": {"path": "synthetic.pt", "size": 123,
                                              "sha256": digest}})
    return gate, counts, digest


def _groups(shape):
    # ALPHA's textual representation is unspecified (1 and 1.0 both valid).
    groups = {}
    for key, value in shape["by_type_alpha"].items():
        kind, alpha = key.rsplit("@", 1)
        identity = kind, float(alpha)
        assert identity not in groups
        groups[identity] = value
    return groups


def _nonzero(counts):
    return {key: value for key, value in counts.items() if value}


@pytest.mark.parametrize("changed", [None, "score_sha256", "n_clusters_peak", "residue_modes"])
def test_report_hand_computed_arithmetic_and_record_identity(work, changed):
    out = work / "report"
    gate, counts, digest = _report_inputs(out, changed)
    _run([sys.executable, "scripts/ruler_checks_report.py", out])
    report = _json(out / "checks.json")
    shape = report["shapes"]["gauss"]
    assert shape["gate"] == gate
    noise = shape["noise_null"]
    assert (noise["k"], noise["n"]) == (4, 518)
    assert noise["rate"] == pytest.approx(4 / 518)
    # Spec's interval is rounded illustration; require exact 95% CP arithmetic.
    assert noise["ci"] == pytest.approx(_cp(4, 518), abs=1e-8)
    groups = _groups(shape)
    assert set(groups) == {("noise", a) for a in (0, 0.5, 1, 2, 3, 4)} | {
        ("mix_same_2", 1), ("mix_same_2", 2), ("part_same_0", 1)}
    for alpha, n, k in ((0, 503, 2), (0.5, 3, 1), (1, 3, 0),
                        (2, 3, 0), (3, 3, 0), (4, 3, 1)):
        group = groups["noise", alpha]
        assert (group["n"], group["k_struct"]) == (n, k)
        assert _nonzero(group["calls"]) == {NONE: n}
    for identity, n, k, median, residues, categories in (
            (("mix_same_2", 1), 4, 2, 0.10, {"K": 2, "E": 2},
             {"context association": 1, NONE: 1, "structural association": 1,
              "composition association": 1}),
            (("mix_same_2", 2), 1, 0, 0.4, {"R": 1}, {NONE: 1}),
            (("part_same_0", 1), 2, 1, 0.09, {"K": 1},
             {"context association": 1, NONE: 1})):
        group = groups[identity]
        assert (group["n"], group["k_struct"]) == (n, k)
        assert group["jsd_q50"] == pytest.approx(median)
        assert _nonzero(group["residue_modes"]) == residues
        assert _nonzero(group["calls"]) == categories
    pooling = report["pooling"]
    assert pooling["records_identical"] is (changed is None)
    assert pooling["n_trained"] == 3
    for pool, count in zip(POOLS, counts):
        assert _nonzero(pooling["trained_calls"][pool]) == count
        assert pooling["trained_share_struct"][pool] == pytest.approx(2 / 3)
    assert report["inputs_sha256"]["synthetic"] == digest
    assert (out / "checks.md").is_file()


def _snapshot(out):
    # Metadata only, as test_shard_scale: skip large results and input symlinks.
    # Directory mtimes catch even transient creation/deletion outside OUT.
    found = {}
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        base = Path(base)
        dirs[:] = [name for name in dirs
                   if name not in ("results", ".git", ".agents", ".codex")
                   and base / name != out and not (base / name).is_symlink()]
        for path in [base, *(base / name for name in files)]:
            if path == out:
                continue
            stat = path.lstat()
            found[str(path.relative_to(ROOT))] = (stat.st_mode, stat.st_size, stat.st_mtime_ns)
    return found


def _counts(value):
    assert isinstance(value, dict)
    assert all(isinstance(key, str) and type(n) is int and n >= 0 for key, n in value.items())


def _number(value):
    assert type(value) in (int, float) and math.isfinite(value)


def _check_schema(report):
    assert isinstance(report["shapes"], dict) and report["shapes"]
    for shape in report["shapes"].values():
        assert isinstance(shape["gate"], dict)
        assert isinstance(shape["by_type_alpha"], dict) and shape["by_type_alpha"]
        for group in _groups(shape).values():
            _counts(group["calls"])
            assert type(group["n"]) is int and group["n"] > 0
            assert type(group["k_struct"]) is int and 0 <= group["k_struct"] <= group["n"]
            assert sum(group["calls"].values()) == group["n"]
            if group["jsd_q50"] is not None:
                _number(group["jsd_q50"])
            _counts(group["residue_modes"])
        noise = shape["noise_null"]
        assert type(noise["n"]) is int and noise["n"] > 0
        assert type(noise["k"]) is int and 0 <= noise["k"] <= noise["n"]
        _number(noise["rate"])
        assert noise["rate"] == pytest.approx(noise["k"] / noise["n"])
        assert isinstance(noise["ci"], list) and len(noise["ci"]) == 2
        for endpoint in noise["ci"]:
            _number(endpoint)
        assert 0 <= noise["ci"][0] <= noise["rate"] <= noise["ci"][1] <= 1
    pooling = report["pooling"]
    assert type(pooling["records_identical"]) is bool
    assert pooling["records_identical"]
    assert type(pooling["n_trained"]) is int and pooling["n_trained"] > 0
    for pool in POOLS:
        _counts(pooling["trained_calls"][pool])
        assert sum(pooling["trained_calls"][pool].values()) == pooling["n_trained"]
        share = pooling["trained_share_struct"][pool]
        _number(share)
        assert 0 <= share <= 1
        assert share == pooling["trained_share_struct"]["trained"]
    assert isinstance(report["inputs_sha256"], dict) and report["inputs_sha256"]
    for name, digest in report["inputs_sha256"].items():
        assert isinstance(name, str) and isinstance(digest, str)
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def test_smoke_once_writes_only_under_out(work):
    out, fs = work / "out", work / "fs"
    out.mkdir()
    fs.mkdir()
    (fs / "keep.txt").write_text("read-only input canary\n")
    # All writable caches/scratch stay under OUT; FS is a sibling canary.
    env = _env(SMOKE="1", JOB="test-ruler-checks", OUT=str(out), FS=str(fs),
               DEADLINE=str(int(time.time()) + 590), WORKERS="1", PYTHON=sys.executable,
               HOME=str(out / "home"), TMPDIR=str(out / "tmp"),
               SCRATCH=str(out / "scratch"), XDG_CACHE_HOME=str(out / "cache"),
               HF_HOME=str(out / "hf"), MPLCONFIGDIR=str(out / "mpl"))
    (out / "tmp").mkdir()
    before = _snapshot(out)
    try:
        _run(["bash", "jobs/ruler_checks/run.sh"], env=env, timeout=600)
    finally:
        assert _snapshot(out) == before, "SMOKE modified worktree outside OUT"
    _check_schema(_json(out / "checks.json"))


def main():
    return 0 if pytest.main([str(Path(__file__)), "-q", *sys.argv[1:]]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
