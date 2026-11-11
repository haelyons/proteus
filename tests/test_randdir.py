"""Frozen specs/randdir.md acceptance A–F; python -m tests.test_randdir exits 0/1.

Real ESM-2 8M runs share module-level caches. No columns implementation is
imported: the CLI artifacts are the public interface under test.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1")

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
EXACT = ("top_val", "top_tok", "res_tok", "res_val", "res_key", "sub_bin", "enr", "ann_tot", "edges")
SUMS = ("sub_sum", "sub_sq", "gram")
SCALARS = ("n_seen", "n_sub", "gram_rows")
PRE_POST = "neurons_pre,neurons_post"
IDENTITY = "neurons_post,randdir_identity"


@lru_cache(None)
def _work():
    # Keep the TemporaryDirectory alive, and all generated files in the worktree.
    return tempfile.TemporaryDirectory(prefix=".randdir-tests-", dir=ROOT)


def _path(name):
    return Path(_work().name) / name


def _command(args, cwd=ROOT, test_sources=False):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1")
    env.pop("PROTEUS_TEST_SOURCES", None)
    env.pop("PYTHONPATH", None)  # Historical subprocess must import its own atlas.
    if test_sources:
        env["PROTEUS_TEST_SOURCES"] = "1"
    proc = subprocess.run([str(x) for x in args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (args, proc.returncode, proc.stdout[-4000:], proc.stderr[-4000:])
    return proc.stdout


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _json(path):
    return json.loads(path.read_text())


@lru_cache(None)
def _historical_tree():
    commits = _command(["git", "log", "--diff-filter=A", "--format=%H", "--", "specs/randdir.md"]).splitlines()
    assert len(commits) == 1, commits
    commit = commits[0]
    tree = _path("spec-commit")
    files = _command(["git", "ls-tree", "-r", "--name-only", commit,
                      "atlas", "scripts/run_atlas.py"]).splitlines()
    assert "scripts/run_atlas.py" in files and "atlas/__init__.py" in files
    for name in files:
        # git show the complete historical package, never the current package.
        content = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=ROOT,
                                 env=os.environ.copy(), capture_output=True, check=True).stdout
        dest = tree / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    return tree


@lru_cache(None)
def _run(name, columns=None, extra=(), historical=False):
    data = ROOT / "data/swissprot_test500.pt"
    assert data.is_file(), data
    out = _path(name)
    args = [sys.executable, "scripts/run_atlas.py", "--model", "8M", "--data", data,
            "--out", out, "--device", "cpu", "--dtype", "fp32", "--seed", "17",
            "--max-seqs", "32", "--max-batches", "3", "--tokens-per-batch", "4096",
            "--pilot-tokens", "512", "--K", "16", "--Kr", "8", "--seed-thresholds",
            "--gram-flush-every", "1"]
    if columns is not None:
        args.extend(["--columns", columns])
    _command(args + list(extra), cwd=_historical_tree() if historical else ROOT,
             test_sources=columns == IDENTITY)
    return out


def _equal(x, y, name):
    assert x.shape == y.shape and x.dtype == y.dtype, name
    if x.is_floating_point():
        nx, ny = torch.isnan(x), torch.isnan(y)
        assert torch.equal(nx, ny), name
        assert torch.equal(x[~nx], y[~ny]), name
    else:
        assert torch.equal(x, y), name


def _compare(a, b, rtol=0):
    for key in EXACT:
        _equal(a[key], b[key], key)
    for key in SCALARS:
        assert a[key] == b[key], key
    for key in SUMS:
        x, y = a[key], b[key]
        assert x.shape == y.shape and x.dtype == y.dtype, key
        if rtol == 0:
            _equal(x, y, key)
        else:
            # Same max-relative normalization as scripts/compare_atlas.py.
            x, y = x.double(), y.double()
            error = float((x - y).abs().max() / y.abs().max().clamp(min=1e-300)) if x.numel() else 0.0
            assert error <= rtol, (key, error)


def _digests(result):
    return {key: hashlib.sha256(result[key].contiguous().numpy().tobytes()).hexdigest()
            for key in EXACT}


@lru_cache(None)
def _multi(columns):
    name = "pre-post" if columns == PRE_POST else "identity"
    return _run(name, columns, ("--pilot-out", str(_path(name + "-pilot.pt"))))


@lru_cache(None)
def _random(repeat=False):
    return _run("random-repeat" if repeat else "random", "randdir:64:0",
                () if repeat else ("--pilot-out", str(_path("random-pilot.pt"))))


@lru_cache(None)
def _shards():
    _random()
    return tuple(_run(f"random-shard-{i}", "randdir:64:0",
                      ("--pilot-from", str(_path("random-pilot.pt")), "--shard", f"{i}/2"))
                 for i in range(2))


def test_a_default_and_explicit_pre_match_spec_commit():
    historical = _run("historical", historical=True)
    expected = _load(historical / "atlas.pt")
    digests = _json(historical / "run.json")["digests"]
    assert digests == _digests(expected)
    for out in (_run("default"), _run("explicit-pre", "neurons_pre")):
        got = _load(out / "atlas.pt")
        _compare(got, expected)
        assert _json(out / "run.json")["digests"] == digests == _digests(got)


def test_b_shared_pre_is_unchanged_and_positive_post_tokens_agree():
    out = _multi(PRE_POST)
    pre = _load(out / "atlas_neurons_pre.pt")
    post = _load(out / "atlas_neurons_post.pt")
    _compare(pre, _load(_run("default") / "atlas.pt"))
    summary = _json(out / "run.json")
    assert summary["sources"]["neurons_pre"]["digests"] == _digests(pre)
    eligible = (pre["top_val"][:, 0] > 0).all(dim=-1)
    assert eligible.any(), "positive-pole comparison must not be vacuous"
    agree = (pre["top_tok"][:, 0] == post["top_tok"][:, 0]).all(dim=-1)
    assert int((agree & eligible).sum()) >= 0.99 * int(eligible.sum())


def test_c_random_repeats_and_shared_pilot_shards_merge():
    plain = _load(_random() / "atlas_randdir_64_0.pt")
    repeat = _load(_random(True) / "atlas_randdir_64_0.pt")
    _compare(repeat, plain)
    for out in (_random(), _random(True)):
        assert _json(out / "run.json")["sources"]["randdir_64_0"]["digests"] == _digests(plain)
    paths = [out / "shard_randdir_64_0.pt" for out in _shards()]
    shards = [_load(path) for path in paths]
    assert all(shard["source"] == "randdir_64_0" for shard in shards)
    assert sum(shard["atlas"]["n_seen"] for shard in shards) == plain["n_seen"]
    out = _path("merged")
    _command([sys.executable, "scripts/merge_atlas.py", "--out", out, "--device", "cpu", *paths])
    _compare(_load(out / "atlas.pt"), plain, rtol=1e-9)


def test_d_identity_matches_post_gelu():
    out = _multi(IDENTITY)
    _compare(_load(out / "atlas_randdir_identity.pt"),
             _load(out / "atlas_neurons_post.pt"), rtol=1e-9)


def test_e_direction_artifacts_follow_seed_norms_and_digest():
    original = _load(_random() / "directions_randdir_64_0.pt")
    repeated = _load(_random(True) / "directions_randdir_64_0.pt")
    changed = _load(_run("random-seed-1", "randdir:64:1") / "directions_randdir_64_1.pt")
    assert original["sha256"] == repeated["sha256"]
    assert len(original["R"]) == len(repeated["R"]) == len(changed["R"]) == 6
    for seed, artifact in ((0, original), (1, changed)):
        assert artifact["seed"] == seed and artifact["M"] == 64
        digest = hashlib.sha256()
        for layer, matrix in enumerate(artifact["R"]):
            assert matrix.device.type == "cpu" and matrix.dtype == torch.float32
            assert matrix.shape == (1280, 64)
            assert torch.allclose(matrix.norm(dim=0), torch.ones(64), atol=1e-5, rtol=0)
            expected = torch.randn(1280, 64, generator=torch.Generator().manual_seed(seed * 1000 + layer),
                                   dtype=torch.float32)
            expected /= expected.norm(dim=0)
            _equal(matrix, expected, f"seed {seed}, layer {layer}")
            digest.update(matrix.contiguous().numpy().tobytes())
        assert artifact["sha256"] == digest.hexdigest()
        summary = _json((_random() if seed == 0 else _run("random-seed-1", "randdir:64:1")) / "run.json")
        assert summary["sources"][f"randdir_64_{seed}"]["directions_sha256"] == artifact["sha256"]
    for layer, (a, b, c) in enumerate(zip(original["R"], repeated["R"], changed["R"])):
        _equal(a, b, f"repeat layer {layer}")
        assert not torch.equal(a, c), layer


@pytest.mark.parametrize("columns", [PRE_POST, IDENTITY])
def test_f_multisource_counts_once_and_reports_redo_differences(columns, record_property):
    out = _multi(columns)
    summary = _json(out / "run.json")
    records = [json.loads(line) for line in (out / "batches.jsonl").read_text().splitlines() if line.strip()]
    assert records
    total = sum(record["n_valid"] for record in records)
    assert total > 0
    names = columns.split(",")
    assert summary["columns"] == columns
    assert set(summary["sources"]) == set(names)
    pilot = _load(_path(("pre-post" if columns == PRE_POST else "identity") + "-pilot.pt"))
    assert set(pilot["sources"]) == set(names)
    assert pilot["args_key"]["columns"] == columns
    assert pilot["n_total"] == total and pilot["n_batches"] == len(records)
    for key in ("plan_sha256", "model"):
        assert key in pilot
    for name in names:
        result = _load(out / f"atlas_{name}.pt")
        source = summary["sources"][name]
        assert source["width"] == 1280
        assert source["n_seen"] == result["n_seen"] == result["n_total_plan"] == total
        if "n_total_plan" in summary:
            assert summary["n_total_plan"] == total
        assert source["digests"] == _digests(result)
        for key in ("stats", "res_seeded_bins", "res_seed_limited_bins", "top_seed_short_groups"):
            assert source[key] == result[key], (name, key)
        for key in ("edges", "top_seed", "res_seed_key"):
            assert key in pilot["sources"][name], (name, key)
        _equal(pilot["sources"][name]["edges"], result["edges"], name + " pilot edges")
    first = summary["sources"][names[0]]
    for key in ("n_seen", "stats", "digests"):
        assert summary[key] == first[key], key
    different = 0
    for record in records:
        redo = record["redo_layers_by_source"]
        assert set(redo) == set(names)
        sets = [set(redo[name]) for name in names]
        assert set(record["redo_layers"]) == set.union(*sets)
        different += any(layers != sets[0] for layers in sets[1:])
    record_property("redo_batches_differ", different)
    print(f"{columns}: redo_layers_by_source differ in {different}/{len(records)} batches")


def main():
    return 0 if pytest.main([str(Path(__file__)), "-q", *sys.argv[1:]]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
