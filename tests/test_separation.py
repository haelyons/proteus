"""Blind arithmetic and invariant checks from specs/separation.md.

Run with pytest or ``python -m tests.test_separation`` (exit status 0/1).
Implementation imports are deferred so collection works before the gate lands.
All inputs are hand-built; no models, production data, or implementation oracles.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from statistics import median

import pytest
import torch

ROOT = Path(__file__).absolute().parents[1]
SOURCES = ("trained", "shuf1", "shuf2", "shuf3")
STATS = ("share_struct", "share_residue_mode", "primary_jsd")


@pytest.fixture
def gate():
    from scripts.separation_gate import ci_separation, cliffs_delta, mann_whitney

    return mann_whitney, cliffs_delta, ci_separation


@pytest.fixture
def work():
    with tempfile.TemporaryDirectory(prefix="separation-", dir=ROOT) as name:
        yield Path(name)


def _pairs(x, y):
    """Enumerate pairs directly, without ranks or a statistics library."""
    greater = sum(a > b for a in x for b in y)
    less = sum(a < b for a in x for b in y)
    ties = sum(a == b for a in x for b in y)
    return greater + ties / 2, (greater - less) / (len(x) * len(y))


def _normal(x, y):
    u, d = _pairs(x, y)
    n1, n2 = len(x), len(y)
    total = n1 + n2
    tie_sum = sum(t ** 3 - t for t in Counter([*x, *y]).values())
    variance = n1 * n2 / 12 * (total + 1 - tie_sum / (total * (total - 1)))
    z = (u - n1 * n2 / 2) / math.sqrt(variance)
    return dict(U=u, z=z, p=math.erfc(abs(z) / math.sqrt(2)), d=d)


def _cp(k, n):
    """Invert exact binomial tails, independently of beta/scipy/share_ci.

    Log probabilities keep the worked example's n=1500 binomial coefficients
    from overflowing. No normal approximation is used for these intervals.
    """
    def root(start, target):
        coefficients = [(j, math.lgamma(n + 1) - math.lgamma(j + 1)
                         - math.lgamma(n - j + 1)) for j in range(start, n + 1)]
        lo, hi = 0.0, 1.0
        for _ in range(80):
            p = (lo + hi) / 2
            tail = math.fsum(math.exp(c + j * math.log(p) + (n - j) * math.log1p(-p))
                             for j, c in coefficients)
            if tail < target:
                lo = p
            else:
                hi = p
        return (lo + hi) / 2

    return (0.0 if k == 0 else root(k, 0.025),
            1.0 if k == n else root(k + 1, 0.975))


def _sign(value):
    return int(value > 0) - int(value < 0)


def _ci_sign(a, b):
    return 1 if a[0] > b[1] else -1 if a[1] < b[0] else 0


def test_worked_example(gate):
    mann_whitney, cliffs_delta, ci_separation = gate
    x, y = [1, 2, 3], [2, 4, 5, 6]
    expected = _normal(x, y)
    assert expected["U"] == 1.5
    assert expected["d"] == -0.75
    assert expected["z"] == pytest.approx(-1.6054, abs=0.0001)
    assert expected["p"] == pytest.approx(0.1084, abs=0.0001)
    assert mann_whitney(x, y) == pytest.approx({k: expected[k] for k in ("U", "z", "p")})
    assert cliffs_delta(x, y) == -0.75
    trained, control = _cp(30, 1500), _cp(60, 1500)
    assert trained == pytest.approx((0.0135, 0.0284), abs=0.0001)
    assert control == pytest.approx((0.0307, 0.0512), abs=0.0001)
    assert ci_separation(trained, control) == -1


@pytest.mark.parametrize("x,y", [
    ([1, 2], [3, 4, 5]),
    ([4, 5, 6], [1, 2]),
    ([0, 0, 1, 3], [0, 2, 2]),
    ([-2, 0, 2], [-2, 0, 2]),
    ([1], [0, 1, 2]),
    ([0, 0, 0], [1, 1, 1, 1]),
])
def test_pair_enumeration_ties_and_symmetry(gate, x, y):
    mann_whitney, cliffs_delta, _ = gate
    expected = _normal(x, y)
    actual, reverse = mann_whitney(x, y), mann_whitney(y, x)
    for key in ("U", "z", "p"):
        assert actual[key] == pytest.approx(expected[key], abs=1e-12)
    assert actual["U"] + reverse["U"] == pytest.approx(len(x) * len(y))
    assert reverse["z"] == pytest.approx(-actual["z"])
    assert reverse["p"] == pytest.approx(actual["p"])
    assert cliffs_delta(x, y) == pytest.approx(expected["d"])
    assert cliffs_delta(y, x) == pytest.approx(-expected["d"])
    assert mann_whitney(x[::-1], y[::-1]) == pytest.approx(actual)


@pytest.mark.parametrize("a,b,expected", [
    ((0.6, 0.8), (0.1, 0.5), 1),
    ((0.1, 0.5), (0.6, 0.8), -1),
    ((0.1, 0.5), (0.5, 0.9), 0),
    ((0.5, 0.9), (0.1, 0.5), 0),
    ((0.2, 0.8), (0.3, 0.7), 0),
    ((0.2, 0.8), (0.2, 0.8), 0),
])
def test_ci_separation_strict_disjointness(gate, a, b, expected):
    assert gate[2](a, b) == expected


def _record(source, unit, structural, residue, jsd, insufficient=False):
    record = dict(source=source, layer=0, unit=unit, col=unit, seed=unit, pole=0,
                  insufficient=insufficient, n_clusters_peak=9 if insufficient else 20,
                  primary="state5", n_desc=2, p_struct=0.005 if structural else 0.005001,
                  p_struct_corr=0.01 if structural else 0.010002, primary_jsd=jsd,
                  p_res=0.001 if residue else 0.5, residue_modes=["K", "E"] if residue else [])
    if insufficient:
        for key in ("primary", "n_desc", "p_struct", "p_struct_corr", "primary_jsd",
                    "p_res", "residue_modes"):
            record[key] = None
    return record


def _write_records(path, records, scores=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(records):
        score = scores[i] if scores is not None else {}
        record["score_sha256"] = hashlib.sha256(
            json.dumps(score, sort_keys=True).encode()).hexdigest()
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    if scores is not None:
        torch.save(scores, path.with_suffix(".scores.pt"))


def _pooling(out, levels):
    records, peaks = {}, {}
    for source, level in zip(SOURCES, levels):
        k = (0, 20, 40)[level]
        rows = [_record(source, i, i < k, i < k,
                        (0.1, 0.4, 0.8)[level] + (0.0, 0.01, 0.03, 0.2)[i % 4])
                for i in range(40)]
        # Four distinct peak maxima, ten repeats each: median is 5, not a mean,
        # a median of all states, or the first state's enrichment.
        maxima = [1.0, 3.0, 7.0, 100.0]
        scores = [{"peak": {"state5": {"log2": [-99.0, maxima[i % 4], -1.0]},
                            "burial3": {"log2": [999.0]}}} for i in range(40)]
        rows.append(_record(source, 40, False, False, None, insufficient=True))
        scores.append({})  # Must be excluded before looking up the primary score.
        _write_records(out / "pooling" / "scores" / f"{source}.jsonl", rows, scores)
        records[source], peaks[source] = rows, median(maxima)
    return records, peaks


def _planted(out, above_stat):
    for shape in ("gauss", "gelu"):
        columns, records = [], []
        parts = [["K", 1], ["E", 1]]

        def add(kind, identities, alpha, structural, residue, jsd, insufficient=False):
            col = len(columns)
            columns.append(dict(col=col, type=kind, alpha=alpha, rep=col, parts=identities))
            records.append(_record(shape, col, structural, residue, jsd, insufficient))

        for kind, identities, values in (
                ("mix_same_2", parts, [0.1, 0.3, 0.5, 1.5]),
                ("part_same_0", parts[:1], [0.1, 0.7]),
                ("part_same_1", parts[1:], [0.3, 0.5])):
            for i, value in enumerate(values):
                veto = shape == "gelu" and kind == "mix_same_2"
                add(kind, identities, 2 + i,
                    i < len(values) // 2 or (veto and above_stat == "share_struct"),
                    i < len(values) // 2 or (veto and above_stat == "share_residue_mode"),
                    (0.5, 0.6, 0.8, 1.7)[i] if veto and above_stat == "primary_jsd" else value)
            # Both exclusions would change the statistic if accidentally included.
            add(kind, identities, 1.999, True, True, 50.0)
            add(kind, identities, 2, False, False, None, insufficient=True)
        # An unrelated high-valued part must not enter parts_max.
        add("part_diff_1", [["E", 2]], 2, True, True, 20.0)
        # This mixture has no eligible observations. Its parts are still present.
        add("mix_diff_2", [["K", 1], ["E", 2]], 1, True, True, 20.0)
        add("mix_diff_2", [["K", 1], ["E", 2]], 3, False, False, None, insufficient=True)
        directory = out / "planted" / shape
        directory.mkdir(parents=True)
        (directory / "columns.json").write_text(json.dumps(columns))
        # Join by unit, not the order of rows in the jsonl file.
        _write_records(directory / "scores" / f"{shape}.jsonl", records[::-1])


def _run(out, work):
    destination = work / "separation.json"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1",
               PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "scripts/separation_gate.py", "--checks", str(out),
                           "--out", str(destination)], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    return json.loads(destination.read_text())


def _check_sources(result, records, peaks):
    for stat in STATS:
        report = result["stats"][stat]
        values, intervals = {}, {}
        for source in SOURCES:
            rows = [r for r in records[source] if not r["insufficient"]]
            summary = report["by_source"][source]
            assert summary["n"] == len(rows)
            if stat == "primary_jsd":
                values[source] = [r["primary_jsd"] for r in rows]
                assert summary["median"] == pytest.approx(median(values[source]))
            else:
                k = sum(r["p_struct_corr"] <= 0.01 if stat == "share_struct"
                        else bool(r["residue_modes"]) for r in rows)
                intervals[source] = _cp(k, len(rows))
                assert summary["k"] == k
                assert summary["share"] == pytest.approx(k / len(rows))
                assert summary["ci"] == pytest.approx(intervals[source], abs=1e-9)
        signs = []
        for control in SOURCES[1:]:
            comparison = report["vs"][control]
            if stat == "primary_jsd":
                expected = _normal(values["trained"], values[control])
                for key in ("U", "z", "p", "d"):
                    assert comparison[key] == pytest.approx(expected[key], abs=1e-10)
                sign = (_sign(expected["d"]) if expected["p"] < 0.01
                        and abs(expected["d"]) >= 0.147 else 0)
            else:
                a, b = intervals["trained"], intervals[control]
                assert comparison["lo_hi_trained"] == pytest.approx(a, abs=1e-9)
                assert comparison["lo_hi_control"] == pytest.approx(b, abs=1e-9)
                sign = _ci_sign(a, b)
            assert comparison["sign"] == sign
            signs.append(sign)
        separates = len(set(signs)) == 1 and signs[0] != 0
        assert report["separates"] is separates
        assert report["direction"] == (signs[0] if separates else 0)
        assert result["qualifies"][stat] is (separates and not report["rates_mixtures_above_parts"])
    assert result["peak_log2_median"] == pytest.approx(peaks)


@pytest.mark.parametrize("levels", [(1, 0, 2, 0), (1, 0, 0, 0), (1, 2, 2, 2), (1, 0, 1, 0)])
@pytest.mark.parametrize("above_stat", [None, *STATS])
def test_hand_built_separation_and_mixture_veto(work, levels, above_stat):
    out = work / "checks"
    records, peaks = _pooling(out, levels)
    _planted(out, above_stat)
    result = _run(out, work)
    _check_sources(result, records, peaks)
    for stat in STATS:
        assert result["stats"][stat]["rates_mixtures_above_parts"] is (stat == above_stat)
        for shape in ("gauss", "gelu"):
            mixture = result["mixtures"][shape]["mix_same_2"][stat]
            above = shape == "gelu" and stat == above_stat
            baseline = 0.4 if stat == "primary_jsd" else 0.5
            expected = (0.7 if stat == "primary_jsd" else 1.0) if above else baseline
            assert mixture["mixture"] == pytest.approx(expected)
            assert mixture["parts_max"] == pytest.approx(baseline)
            assert mixture["above"] is above
            unavailable = result["mixtures"][shape]["mix_diff_2"][stat]
            assert unavailable["mixture"] is None
            assert unavailable["above"] is False


@pytest.mark.parametrize("n,k", [(4, 4), (400, 240), (400, 280)])
def test_jsd_requires_both_p_and_effect_size(work, n, k):
    out = work / "checks"
    records, peaks = {}, {}
    for source in SOURCES:
        successes = k if source == "trained" else n // 2
        rows = [_record(source, i, False, False, float(i < successes)) for i in range(n)]
        scores = [{"peak": {"state5": {"log2": [1.0]}}} for _ in rows]
        _write_records(out / "pooling" / "scores" / f"{source}.jsonl", rows, scores)
        records[source], peaks[source] = rows, 1.0
    _planted(out, None)
    result = _run(out, work)
    _check_sources(result, records, peaks)
    expected = _normal([r["primary_jsd"] for r in records["trained"]],
                       [r["primary_jsd"] for r in records["shuf1"]])
    # Respectively: large effect without significance; significant small effect;
    # both criteria pass. This distinguishes the AND from either criterion alone.
    assert (expected["p"] < 0.01) is (n != 4)
    assert (abs(expected["d"]) >= 0.147) is (n == 4 or k == 280)
    assert result["qualifies"]["primary_jsd"] is (n == 400 and k == 280)


def test_mixture_without_eligible_parts_is_not_above(work):
    out = work / "checks"
    _pooling(out, (1, 0, 0, 0))
    directory = out / "planted" / "gauss"
    directory.mkdir(parents=True)
    parts = [["K", 1], ["E", 1]]
    columns = [dict(col=0, type="mix_same_2", alpha=2, rep=0, parts=parts),
               dict(col=1, type="part_same_0", alpha=1, rep=0, parts=parts[:1]),
               dict(col=2, type="part_same_1", alpha=2, rep=0, parts=parts[1:])]
    (directory / "columns.json").write_text(json.dumps(columns))
    rows = [_record("gauss", 0, True, True, 0.8),
            _record("gauss", 1, False, False, 0.1),
            _record("gauss", 2, False, False, None, insufficient=True)]
    _write_records(directory / "scores" / "gauss.jsonl", rows)
    result = _run(out, work)
    for stat in STATS:
        mixture = result["mixtures"]["gauss"]["mix_same_2"][stat]
        assert mixture["mixture"] == pytest.approx(0.8 if stat == "primary_jsd" else 1.0)
        assert mixture["parts_max"] is None
        assert mixture["above"] is False
        assert result["stats"][stat]["rates_mixtures_above_parts"] is False
        assert result["qualifies"][stat] is True


@pytest.mark.parametrize("status", [0, 1, 2, 3, 4, 5])
def test_entry_point_normalizes_status(monkeypatch, status):
    monkeypatch.setattr(pytest, "main", lambda args: status)
    assert main() == (0 if status == 0 else 1)


def main():
    return 0 if pytest.main([str(Path(__file__)), "-q", *sys.argv[1:]]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
