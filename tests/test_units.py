"""Blind acceptance tests from frozen specs/units.md, sections 4 and 5.

The spec does not name arithmetic helpers or freeze the card check schema.
These tests use this minimal adapter contract in scripts/units_study.py:

* profile_distance(p, q): mean base-2 JSD of three probability vectors.
* rank_pass(own, controls, higher=False): strict rank against 20 controls.
* spearman(x, y): average ranks for ties.
* bootstrap_diff(target, control, paired=True): [lo, hi], default 2,000
  draws, seed 0, 95%; paired=False samples the two position sets independently.

Profiles are lists [residue, ss3, burial]. Cards expose `checks` with keys
profile/write_steer/removal, each with a boolean `pass` (null if insufficient).
The removal check also contains `resample`, with its secondary boolean `pass`.
Only these adapters need changing if the eventual public helper names differ.
No implementation imports or job execution happen during collection.
"""
from __future__ import annotations

import importlib.util
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

ROOT = Path(__file__).absolute().parents[1]
CHECKS = {"profile", "write_steer", "removal"}
PASS_KEYS = CHECKS | {"removal_resample"}


@pytest.fixture
def study():
    spec = importlib.util.spec_from_file_location("units_study", ROOT / "scripts/units_study.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def _profile(first, ss3=None, burial=None):
    return [first, ss3 or [1, 0, 0, 0], burial or [1, 0, 0, 0]]


def test_jsd_base_two_mean_and_zero_bins(study):
    p = _profile([1, 0] + [0] * 18)
    q = _profile([0, 1] + [0] * 18)
    uniform = _profile([0.5, 0.5] + [0] * 18)
    assert study.profile_distance(p, p) == pytest.approx(0, abs=1e-12)
    assert study.profile_distance(p, q) == pytest.approx(1 / 3)
    assert study.profile_distance(q, p) == pytest.approx(1 / 3)
    # JSD(delta, uniform on two bins) = H(3/4,1/4) - 1/2.
    expected = (-0.75 * math.log2(0.75) - 0.25 * math.log2(0.25) - 0.5) / 3
    assert study.profile_distance(p, uniform) == pytest.approx(expected)
    # All three distributions have equal weight, despite different bin counts.
    disjoint = _profile([0, 1] + [0] * 18, [0, 1, 0, 0], [0, 0, 0, 1])
    assert study.profile_distance(p, disjoint) == pytest.approx(1)


@pytest.mark.parametrize("higher", [False, True])
def test_rank_requires_strictly_beating_all_twenty(study, higher):
    own, worse, better = (0.5, 0.25, 0.75) if higher else (0.5, 0.75, 0.25)
    assert study.rank_pass(own, [worse] * 20, higher=higher)
    # One tie gives rank p >= 2/21, so it cannot pass p < 0.05.
    for index in range(20):
        for rival in (own, better):
            controls = [worse] * 20
            controls[index] = rival
            assert not study.rank_pass(own, controls, higher=higher)


def test_profile_distance_drives_rank_in_correct_direction(study):
    heldout = _profile([1, 0] + [0] * 18)
    own = _profile([0.75, 0.25] + [0] * 18)
    others = [_profile([0, 1] + [0] * 18) for _ in range(20)]
    distance = study.profile_distance(heldout, own)
    controls = [study.profile_distance(heldout, other) for other in others]
    assert study.rank_pass(distance, controls, higher=False)
    controls[-1] = study.profile_distance(heldout, heldout)
    assert not study.rank_pass(distance, controls, higher=False)


@pytest.mark.parametrize("x,y,expected", [
    ([1, 2, 2, 4], [4, 1, 1, 2], -1 / 3),
    ([1, 1, 3, 4], [2, 3, 3, 1], -0.5),
    ([0, 0, 2, 3], [4, 4, 8, 9], 1),
    ([0, 0, 2, 3], [9, 9, 4, 0], -1),
])
def test_spearman_uses_average_ranks_for_ties(study, x, y, expected):
    # First two examples have ranks [1,2.5,2.5,4]/[4,1.5,1.5,3]
    # and [1.5,1.5,3,4]/[2,3.5,3.5,1], respectively.
    assert study.spearman(x, y) == pytest.approx(expected, abs=1e-12)
    assert study.spearman(y, x) == pytest.approx(expected, abs=1e-12)


def test_write_steering_rank_uses_other_writes_with_target_steering(study):
    steering = list(range(20))
    direct = list(range(20))
    controls = [list(reversed(steering)) for _ in range(20)]
    own = study.spearman(direct, steering)
    correlations = [study.spearman(other, steering) for other in controls]
    assert study.rank_pass(own, correlations, higher=True)
    correlations[7] = study.spearman(direct, steering)
    assert not study.rank_pass(own, correlations, higher=True)


@pytest.mark.parametrize("paired", [False, True])
def test_bootstrap_equal_control_never_passes(study, paired):
    target = [0, 1, 7, 2, 8, 3, 0, 4, 6, 2]
    interval = study.bootstrap_diff(target, target, paired=paired)
    assert len(interval) == 2
    assert interval[0] <= 0 <= interval[1]
    if paired:
        assert list(interval) == [0, 0]
    assert not interval[0] > 0


@pytest.mark.parametrize("paired", [False, True])
def test_bootstrap_target_above_every_control_passes(study, paired):
    target = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    controls = [[0] * 10, [1, 2, 3, 4, 5, 0, 1, 2, 3, 4], [7] * 10]
    assert all(study.bootstrap_diff(target, c, paired=paired)[0] > 0 for c in controls)


def test_bootstrap_joint_versus_independent_positions(study):
    # Paired differences are always 1. Independent means range from -99 to 101;
    # each extreme has probability 1/16, exceeding the 2.5% interval tails.
    target, control = [1, 101], [0, 100]
    assert list(study.bootstrap_diff(target, control, paired=True)) == [1, 1]
    independent = study.bootstrap_diff(target, control, paired=False)
    assert list(independent) == [-99, 101]
    assert list(study.bootstrap_diff(target, control, paired=False)) == list(independent)
    assert not independent[0] > 0
    # One failed control vetoes check 3, even when the other two pass.
    for failing in range(3):
        controls = [[-10, -10] for _ in range(3)]
        controls[failing] = target
        intervals = [study.bootstrap_diff(target, c, paired=i < 2)
                     for i, c in enumerate(controls)]
        assert not all(ci[0] > 0 for ci in intervals)


def test_bootstrap_95_percent_tails_on_nonconstant_differences(study):
    # Four jointly sampled positions give Binomial(4, 1/4)/4 differences.
    # Exact 2.5%/97.5% quantiles are 0 and 3/4; the maximum has mass 1/256.
    interval = study.bootstrap_diff([0, 0, 0, 1], [0, 0, 0, 0], paired=True)
    assert list(interval) == [0, 0.75]


def test_worked_example_clopper_pearson_interval():
    from atlas.compare import share_ci

    assert 23 / (150 - 12) == pytest.approx(0.16667, abs=5e-6)
    assert share_ci(23, 138) == pytest.approx([0.10868, 0.23951], abs=5e-6)


def test_quantile_linear_worked_example(study):
    import torch

    x = torch.tensor([4., 1., 3., 2.], dtype=torch.float32)
    q = torch.tensor([1 / 3, 2 / 3], dtype=x.dtype)
    actual = study.quantile_linear(x, q)
    expected = torch.tensor([2., 3.], dtype=x.dtype)
    assert actual.shape == q.shape
    assert actual.dtype == x.dtype
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("n", [1, 2, 7, 257, 4096, 2 ** 24])
@pytest.mark.parametrize("ties", [False, True])
def test_quantile_linear_matches_torch_bitwise(study, dtype, n, ties):
    import torch

    # Resolve the helper before allocating large inputs when it is absent.
    quantile_linear = study.quantile_linear
    x = torch.randn(n, dtype=getattr(torch, dtype),
                    generator=torch.Generator().manual_seed(20261112))
    if ties:
        x = x.round()
    q = torch.tensor([0., 1 / 3, 0.17, 0.5, 0.73, 2 / 3, 1.], dtype=x.dtype)
    expected = torch.quantile(x, q)
    actual = quantile_linear(x, q)
    assert actual.shape == q.shape
    assert actual.dtype == x.dtype
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_quantile_linear_above_torch_size_limit(study):
    import torch

    quantile_linear = study.quantile_linear
    n = 2 ** 24 + 7
    x = torch.randn(n, dtype=torch.float32,
                    generator=torch.Generator().manual_seed(20261112))
    # Include a fractional rank near zero even at this float32 input size.
    q = torch.tensor([0., 0.5 / (n - 1), 1e-6, 0.1234567, 0.5, 0.75, 1.],
                     dtype=x.dtype)
    ranks = q.double() * (n - 1)
    floors, ceils = ranks.floor(), ranks.ceil()
    lower = torch.stack([torch.kthvalue(x, int(rank) + 1).values for rank in floors])
    upper = torch.stack([torch.kthvalue(x, int(rank) + 1).values for rank in ceils])
    expected = torch.lerp(lower, upper, (ranks - floors).to(x.dtype))
    actual = quantile_linear(x, q)
    assert actual.shape == q.shape
    assert actual.dtype == x.dtype
    assert torch.all(lower <= actual)
    assert torch.all(actual <= upper)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def _json(path):
    return json.loads(path.read_text())


def _check_pass_counts(summary, cards):
    from atlas.compare import share_ci

    assert summary["n"] == len(cards)
    assert summary["insufficient"] == sum(card["insufficient"] for card in cards)
    assert set(summary["pass"]) == PASS_KEYS
    for name, report in summary["pass"].items():
        sufficient = [card for card in cards
                      if name == "write_steer" or not card["insufficient"]]
        values = [(card["checks"]["removal"]["resample"] if name == "removal_resample"
                   else card["checks"][name])["pass"] for card in sufficient]
        assert all(type(value) is bool for value in values)
        k, n = sum(values), len(values)
        assert {"k", "n", "rate", "ci"} <= set(report)
        assert (report["k"], report["n"]) == (k, n)
        if n:
            assert report["rate"] == pytest.approx(k / n, abs=5e-6)
        else:
            # The spec leaves the empty rate undefined; JSON null or 0 is accepted.
            assert report["rate"] in (None, 0)
        assert report["ci"] == pytest.approx(share_ci(k, n), abs=5e-6)


def test_units_smoke_end_to_end():
    budget = float(os.environ.get("UNITS_SMOKE_BUDGET", "1800"))
    assert budget > 0
    data = ROOT / "data/swissprot_test500.pt"
    assert data.is_file(), data
    with tempfile.TemporaryDirectory(prefix=".units-smoke-", dir=ROOT) as directory:
        work = Path(directory)
        out, fs = work / "out", work / "fs"
        for path in (out, fs / "cache", work / "tmp", work / "cache"):
            path.mkdir(parents=True)
        (fs / "cache/swissprot_2026_03.pt").symlink_to(data)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1",
                   PYTHONDONTWRITEBYTECODE="1", SMOKE="1", JOB="test-units-8m",
                   OUT=str(out), FS=str(fs), CALIB=str(out / "smoke_inputs"), WORKERS="1",
                   DEADLINE=str(int(time.time() + budget)), PYTHON=sys.executable,
                   SMOKE_DATA=str(data), TMPDIR=str(work / "tmp"),
                   MPLCONFIGDIR=str(work / "cache/mpl"),
                   PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
        result = subprocess.run(["bash", "jobs/units_8m/run.sh"], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=budget)
        assert result.returncode == 0, (result.returncode, result.stdout[-6000:],
                                        result.stderr[-6000:])
        report = _json(out / "units.json")
        assert {"populations", "seeds", "positions"} <= set(report)
        assert isinstance(report["seeds"], dict) and report["seeds"]
        assert report["positions"] == {"peak_max": 60, "random": 20, "margin": 3}
        assert set(report["populations"]) == {"neurons", "directions"}
        sample = _json(out / "sample.json")
        assert {"neurons", "directions", "directions_sha256",
                "calib_directions_sha256"} <= set(sample)
        for population in ("neurons", "directions"):
            chosen = [tuple(pair) for pair in sample[population]]
            assert len(set(chosen)) == len(chosen) == 12
            assert Counter(l for l, _ in chosen) == {l: 2 for l in range(6)}
            paths = list((out / "cards").glob(f"{population}_*.json"))
            assert {path.stem for path in paths} == {
                f"{population}_{layer}_{index}" for layer, index in chosen}
            cards = {(layer, index): _json(out / "cards" /
                                         f"{population}_{layer}_{index}.json")
                     for layer, index in chosen}
            for (layer, index), card in cards.items():
                assert set(card["checks"]) == CHECKS
                assert type(card["insufficient"]) is bool
                assert (out / "cards" / f"{population}_{layer}_{index}.md").read_text().strip()
            summary = report["populations"][population]
            assert {"n", "insufficient", "pass", "by_layer"} <= set(summary)
            _check_pass_counts(summary, list(cards.values()))
            assert set(summary["by_layer"]) == {str(l) for l in range(6)}
            for layer in range(6):
                _check_pass_counts(summary["by_layer"][str(layer)],
                                   [card for (l, _), card in cards.items() if l == layer])
