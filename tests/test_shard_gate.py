"""Unit 3 acceptance I–L from specs/shard.md (real ESM-2 8M, CPU).

Run with pytest or ``python -m tests.test_shard_gate``. Model runs are cached
per module; no implementation helpers serve as comparison oracles.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).absolute().parents[1]
EXACT = ("top_val", "top_tok", "res_tok", "res_val", "res_key", "sub_bin", "enr", "ann_tot", "edges")
CLOSE = ("sub_sum", "sub_sq", "gram")
SCALARS = ("n_seen", "n_sub", "gram_rows")
# Keep TemporaryDirectory alive until module teardown, including between pytest functions.
_TMP = tempfile.TemporaryDirectory(prefix="proteus-shard-gate-", dir=ROOT)
WORK = Path(_TMP.name)


def _env(**extra):
    return dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1", **extra)


def _run(args, expected=0, env=None):
    p = subprocess.run([str(x) for x in args], cwd=ROOT, env=env or _env(),
                       capture_output=True, text=True, timeout=540)
    assert p.returncode == expected, (args, p.returncode, p.stdout, p.stderr)
    return p


def _data():
    path = Path(os.environ.get("PROTEUS_TEST_DATA", "data/swissprot_test500.pt"))
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        pytest.skip(f"ESM-2 gate needs dataset: {path} (set PROTEUS_TEST_DATA)")
    return path


def _plan_args():
    return ["--model", "8M", "--data", str(_data()), "--device", "cpu", "--dtype", "fp32",
            "--seed", "0", "--tokens-per-batch", "4096", "--window", "8192",
            "--max-seqs", "40", "--max-batches", "2"]


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _exact(x, y):
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    if x.is_floating_point():
        nx, ny = torch.isnan(x), torch.isnan(y)
        return torch.equal(nx, ny) and torch.equal(x[~nx], y[~ny])
    return torch.equal(x, y)


def _relative(x, y):
    # Same normalization as the existing public compare CLI, computed independently.
    x, y = x.double(), y.double()
    return float((x - y).abs().max() / y.abs().max().clamp(min=1e-300))


def _assert_atlases(a, b, rtol):
    for key in EXACT:
        assert _exact(a[key], b[key]), key
    for key in CLOSE:
        assert a[key].shape == b[key].shape, key
        assert _relative(a[key], b[key]) <= rtol, key
    for key in SCALARS:
        assert a[key] == b[key], key


@lru_cache(maxsize=None)
def _reference():
    out = WORK / "reference"
    _run([sys.executable, "scripts/run_atlas.py", *_plan_args(), "--out", out,
          "--pilot-tokens", "4096", "--K", "16", "--Kr", "8", "--seed-thresholds"])
    assert (out / "atlas.pt").is_file()
    assert (out / "run.json").is_file()
    return out


@lru_cache(maxsize=None)
def _perturbed():
    src = _load(_reference() / "atlas.pt")
    changed = dict(src, sub_sum=src["sub_sum"].clone() * 1.001)
    # Guard against an empty/zero sum making the intended rejection vacuous.
    assert 0.0009 < _relative(changed["sub_sum"], src["sub_sum"]) < 0.0011
    for key in EXACT:
        assert _exact(src[key], changed[key]), key
    path = WORK / "sum-perturbed.pt"
    torch.save(changed, path)
    return path


def _compare(a, b=None, *, digests=None, rtol=None, expected=0, tag):
    out = WORK / f"compare-{tag}.json"
    args = [sys.executable, "scripts/compare_atlas.py", a]
    args += ["--digests", digests] if digests is not None else [b]
    if rtol is not None:
        args += ["--rtol", str(rtol)]
    _run([*args, "--out", out], expected=expected)
    report = json.loads(out.read_text())
    assert report["ok"] is (expected == 0)
    assert report["rtol"] == (1e-5 if rtol is None else rtol)
    assert report["ok"] == (report["exact_all"] and report["scalars_all"] and
                            (True if digests is not None else report["close_all"]))
    return report


def test_i_compare_exact_sums_and_run_digests():
    out = _reference()
    atlas = out / "atlas.pt"
    a = _load(atlas)
    run = json.loads((out / "run.json").read_text())
    for key in EXACT:
        want = hashlib.sha256(a[key].contiguous().numpy().tobytes()).hexdigest()
        assert run["digests"][key] == want, key
    same = _compare(atlas, atlas, tag="self")
    for key in EXACT:
        assert same["fields"][key]["exact"] is True
    for key in CLOSE:
        assert same["fields"][key]["ok"] is True
        assert same["fields"][key]["max_rel_diff"] == 0
    digest = _compare(atlas, digests=out / "run.json", tag="digests")
    assert digest["close_checked"] is False
    for key in EXACT:
        assert digest["fields"][key] == {"exact": True, "a": run["digests"][key], "b": run["digests"][key]}

    changed = dict(a, top_tok=a["top_tok"].clone())
    changed["top_tok"].view(-1)[0] += 1
    assert int((changed["top_tok"] != a["top_tok"]).sum()) == 1
    bad_tok = WORK / "bad-token.pt"
    torch.save(changed, bad_tok)
    report = _compare(bad_tok, atlas, expected=1, tag="token")
    assert report["exact_all"] is False
    assert report["fields"]["top_tok"]["exact"] is False
    report = _compare(_perturbed(), atlas, expected=1, tag="sum")
    assert report["close_all"] is False
    assert report["fields"]["sub_sum"]["ok"] is False
    assert report["fields"]["sub_sum"]["max_rel_diff"] == _relative(_load(_perturbed())["sub_sum"], a["sub_sum"])

    bad_run = copy.deepcopy(run)
    old = bad_run["digests"]["top_tok"]
    bad_run["digests"]["top_tok"] = ("0" if old[0] != "0" else "1") + old[1:]
    bad_json = WORK / "bad-digest.json"
    bad_json.write_text(json.dumps(bad_run))
    report = _compare(atlas, digests=bad_json, expected=1, tag="bad-digest")
    assert report["fields"]["top_tok"]["exact"] is False

    # Missing fields must fail in either direction, including float sums and scalars.
    for key in ("top_tok", "sub_sum", "n_seen"):
        missing = dict(a)
        del missing[key]
        path = WORK / f"missing-{key}.pt"
        torch.save(missing, path)
        for left, right, side in ((path, atlas, "a"), (atlas, path, "b")):
            report = _compare(left, right, expected=1, tag=f"missing-{key}-{side}")
            assert key in report["missing"]
    for key in ("n_seen", "gram_rows"):
        bad_run = copy.deepcopy(run)
        bad_run[key] = a[key] + 1
        bad_json.write_text(json.dumps(bad_run))
        report = _compare(atlas, digests=bad_json, expected=1, tag=f"scalar-{key}")
        assert report["scalars_all"] is False
    missing_run = copy.deepcopy(run)
    del missing_run["digests"]["top_tok"]
    bad_json.write_text(json.dumps(missing_run))
    report = _compare(atlas, digests=bad_json, expected=1, tag="missing-digest")
    assert "top_tok" in report["missing"]

    # A real NaN must match itself without equating it to a finite sentinel.
    nan_a = dict(a, res_val=a["res_val"].clone())
    nan_a["res_val"].view(-1)[0] = float("nan")
    nan_b = copy.deepcopy(nan_a)
    assert _exact(nan_a["res_val"], nan_b["res_val"])
    pa, pb = WORK / "nan-a.pt", WORK / "nan-b.pt"
    torch.save(nan_a, pa)
    torch.save(nan_b, pb)
    _compare(pa, pb, tag="nan-equal")


def test_j_rtol_controls_float_sum_gate():
    atlas = _reference() / "atlas.pt"
    perturbed = _perturbed()
    _compare(perturbed, atlas, expected=1, tag="rtol-default")
    report = _compare(perturbed, atlas, rtol=1e-2, tag="rtol-loose")
    assert report["close_all"] is True
    assert report["fields"]["sub_sum"]["ok"] is True
    assert report["fields"]["sub_sum"]["max_rel_diff"] == _relative(_load(perturbed)["sub_sum"], _load(atlas)["sub_sum"])


def _digest_leaves(value, path=()):
    # Spec defines digest content but leaves JSON nesting unspecified.
    if isinstance(value, dict):
        for key, child in value.items():
            if key != "plan_sha256":
                yield from _digest_leaves(child, path + (key,))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from _digest_leaves(child, path + (i,))
    elif isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
        yield path, value


@lru_cache(maxsize=None)
def _batch_runs():
    paths = [WORK / "batch-a.json", WORK / "batch-b.json"]
    for path in paths:
        _run([sys.executable, "scripts/batch_digest.py", *_plan_args(), "--batch", "0", "1", "--out", path])
        assert path.is_file()
    return paths


def test_k_cpu_batch_digests_repeat_and_detect_edit():
    pa, pb = _batch_runs()
    a, b = json.loads(pa.read_text()), json.loads(pb.read_text())
    assert re.fullmatch(r"[0-9a-f]{64}", a["plan_sha256"])
    assert a["plan_sha256"] == b["plan_sha256"]
    da, db = dict(_digest_leaves(a)), dict(_digest_leaves(b))
    assert da, "batch_digest wrote no activation digests"
    assert da == db, "independent CPU forwards differ"
    _run([sys.executable, "scripts/batch_digest.py", "--compare", pa, pb])
    edited = copy.deepcopy(b)
    path, old = next(iter(db.items()))
    parent = edited
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = ("0" if old[0] != "0" else "1") + old[1:]
    assert sum(dict(_digest_leaves(edited))[key] != value for key, value in db.items()) == 1
    bad = WORK / "batch-edited.json"
    bad.write_text(json.dumps(edited))
    # Spec requires non-zero, not a particular refusal code.
    p = subprocess.run([sys.executable, "scripts/batch_digest.py", "--compare", pa, bad],
                       cwd=ROOT, env=_env(), capture_output=True, text=True, timeout=30)
    assert p.returncode != 0, (p.stdout, p.stderr)


@lru_cache(maxsize=None)
def _smoke():
    data = _data()
    out = WORK / "smoke"
    out.mkdir()
    # Other jobs use SMOKE_DATA; PROTEUS_TEST_DATA is the test convention.
    # Keep FS local so a job cannot accidentally use a production filesystem.
    fs = WORK / "fs"
    (fs / "cache").mkdir(parents=True)
    (fs / "cache" / "swissprot_2026_03.pt").symlink_to(data)
    env = _env(SMOKE="1", JOB="test-shard-gate", OUT=str(out), FS=str(fs),
               DEADLINE=str(int(time.time()) + 500), NSHARDS="2",
               SMOKE_DATA=str(data), PROTEUS_TEST_DATA=str(data),
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    _run(["bash", "jobs/shard_gate/run.sh"], env=env)
    return out


def test_l_smoke_shard_gate_end_to_end():
    out = _smoke()
    report = json.loads((out / "gate_compare.json").read_text())
    assert report["ok"] is True
    assert report["exact_all"] and report["close_all"] and report["scalars_all"]
    assert json.loads((out / "timeline.json").read_text()), "empty timeline"
    # Directory names are unspecified. Inspect both output atlases independently.
    paths = sorted(out.rglob("atlas.pt"))
    assert len(paths) == 2, paths
    _assert_atlases(_load(paths[0]), _load(paths[1]), rtol=1e-9)
    shards = sorted(out.rglob("shard.pt"))
    assert len(shards) == 2, shards
    assert sorted(_load(path)["shard"] for path in shards) == [[0, 2], [1, 2]]
    merge_reports = list(out.rglob("merge.json"))
    assert len(merge_reports) == 1
    assert json.loads(merge_reports[0].read_text())["digests"]


def main():
    # Normalize pytest's status codes (including interruption/collection errors) to 0/1.
    return 0 if pytest.main([str(Path(__file__)), "-q", *sys.argv[1:]]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
