"""Unit 2 acceptance F-H: real-model shard merging, order, and refusals.

Run with pytest or python -m tests.test_shard_merge. Forward runs are cached
per module; all generated files live in a temporary directory.
"""
from __future__ import annotations

import functools
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
EXACT = ("top_val", "top_tok", "res_tok", "res_val", "res_key", "sub_bin", "enr", "ann_tot", "edges")
SUMS = ("sub_sum", "sub_sq", "gram")
SCALARS = ("n_seen", "n_sub", "gram_rows")
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1")


@functools.lru_cache(None)
def _workspace():
    data = Path(os.environ.get("PROTEUS_TEST_DATA", "data/swissprot_test500.pt"))
    if not data.is_absolute():
        data = ROOT / data
    if not data.is_file():
        pytest.skip(f"real-model shard merge tests need dataset: {data}")
    # Retain the owner until module teardown so cached paths remain valid.
    tmp = tempfile.TemporaryDirectory(prefix="proteus-shard-merge-")
    return tmp, Path(tmp.name), data


@functools.lru_cache(None)
def _command(args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=ENV,
                          capture_output=True, text=True)


def _success(args):
    proc = _command(tuple(str(a) for a in args))
    assert proc.returncode == 0, f"{args}\n{proc.stdout}\n{proc.stderr}"


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@functools.lru_cache(None)
def _run_atlas(extra):
    _, root, data = _workspace()
    out = root / f"run-{_run_atlas.cache_info().misses}"
    _success(("scripts/run_atlas.py", "--model", "8M", "--data", data,
              "--device", "cpu", "--dtype", "fp32", "--tokens-per-batch", "4096",
              "--window", "8", "--pilot-tokens", "4096", "--K", "16", "--Kr", "8",
              "--max-seqs", "48", "--max-batches", "8", "--seed", "11",
              # Flush each batch: F compares fp64 sums of identical per-batch
              # Gram terms, not different groupings of fp32 accumulation.
              "--gram-flush-every", "1", "--seed-thresholds", "--out", out, *extra))
    return out


@functools.lru_cache(None)
def _pilot():
    _, root, _ = _workspace()
    path = root / "pilot.pt"
    _run_atlas(("--pilot-out", str(path), "--pilot-only"))
    assert path.is_file(), "pilot artifact missing"
    return str(path)


@functools.lru_cache(None)
def _single():
    out = _run_atlas(("--pilot-from", _pilot()))
    assert (out / "atlas.pt").is_file()
    return out / "atlas.pt"


@functools.lru_cache(None)
def _shards(n):
    paths = []
    for i in range(n):
        out = _run_atlas(("--pilot-from", _pilot(), "--shard", f"{i}/{n}"))
        assert (out / "shard.pt").is_file()
        paths.append(out / "shard.pt")
    return tuple(paths)


@functools.lru_cache(None)
def _merge(n, order):
    _, root, _ = _workspace()
    out = root / (f"merge-{n}-" + "-".join(map(str, order)))
    shards = _shards(n)
    _success(("scripts/merge_atlas.py", "--out", out, "--device", "cpu",
              *(shards[i] for i in order)))
    assert (out / "atlas.pt").is_file()
    assert (out / "merge.json").is_file()
    return out / "atlas.pt"


def _exact(x, y, label):
    assert x.shape == y.shape and x.dtype == y.dtype, label
    if x.is_floating_point():
        # Empty reservoir slots contain NaNs. Check their locations separately
        # so a real value cannot compare equal to a replaced NaN.
        xn, yn = torch.isnan(x), torch.isnan(y)
        assert torch.equal(xn, yn), f"{label}: NaN locations"
        x, y = x.masked_fill(xn, 0), y.masked_fill(yn, 0)
    assert torch.equal(x, y), label


def _compare(got, want, *, bitwise_sums):
    for key in EXACT:
        assert key in got and key in want, f"missing exact field: {key}"
        _exact(got[key], want[key], key)
    for key in SCALARS:
        assert key in got and key in want, f"missing scalar: {key}"
        assert got[key] == want[key], key
    for key in SUMS:
        assert key in got and key in want, f"missing float sum: {key}"
        x, y = got[key], want[key]
        assert x.shape == y.shape and x.dtype == y.dtype, key
        if bitwise_sums:
            _exact(x, y, key)
        else:
            x, y = x.double(), y.double()
            rel = float((x - y).abs().max() / y.abs().max().clamp(min=1e-300))
            assert rel <= 1e-9, f"{key}: max relative difference {rel} > 1e-9"


def test_f_merged_shards_equal_single():
    single = _load(_single())
    assert single["n_seen"] > 0 and single["n_sub"] > 0 and single["gram_rows"] > 0
    # Several whole batches exercise both within-shard and between-shard sums.
    assert single["n_batches"] > 4
    for n in (1, 2, 4):
        merged = _load(_merge(n, tuple(range(n))))
        _compare(merged, single, bitwise_sums=False)


def test_g_command_line_order_is_bitwise_identical():
    sorted_result = _load(_merge(4, (0, 1, 2, 3)))
    for order in ((3, 2, 1, 0), (2, 0, 3, 1)):
        _compare(_load(_merge(4, order)), sorted_result, bitwise_sums=True)


def test_h_invalid_shards_are_refused():
    shards = _shards(2)
    _, root, _ = _workspace()
    changed = root / "changed-shard.pt"

    def refuse(name, inputs, allow_partial=False):
        out = root / f"refused-{name}"
        args = ("scripts/merge_atlas.py", "--out", out, "--device", "cpu", *inputs)
        if allow_partial:
            args += ("--allow-partial",)
        proc = _command(tuple(map(str, args)))
        assert proc.returncode != 0, f"{name}: unexpectedly accepted\n{proc.stdout}\n{proc.stderr}"
        assert (proc.stdout + proc.stderr).strip(), f"{name}: no refusal message"
        assert not (out / "atlas.pt").exists(), f"{name}: wrote atlas.pt"
        assert not out.exists() or not any(out.iterdir()), f"{name}: wrote output on refusal"

    refuse("duplicate-index", (shards[0], shards[0]))
    refuse("duplicate-index-partial", (shards[0], shards[0]), allow_partial=True)
    refuse("missing-index", (shards[0],))
    partial = root / "allowed-missing-index"
    _success(("scripts/merge_atlas.py", "--out", partial, "--device", "cpu",
              "--allow-partial", shards[0]))
    assert (partial / "atlas.pt").is_file()

    # Fresh state for every mutation; all other metadata and tensors stay valid.
    for field in ("N", "plan_sha256", "n_batches_plan", "n_total_plan", "args_key",
                  "batches_done", "edges", "top_seed", "res_seed"):
        st = _load(shards[1])
        if field == "N":
            st["shard"] = [1, 3]
        elif field == "plan_sha256":
            digest = st[field]
            st[field] = ("0" if digest[0] != "0" else "1") + digest[1:]
        elif field in ("n_batches_plan", "n_total_plan"):
            st[field] += 1
        elif field == "args_key":
            st[field]["seed"] += 1
        elif field == "batches_done":
            st[field] -= 1
        else:
            key = "E" if field == "edges" else field
            tensor = st["atlas"][key]
            flat = tensor.reshape(-1)
            if tensor.is_floating_point():
                # nextafter changes even +/-inf seeds without depending on
                # their magnitude or on which groups the pilot populated.
                flat[0] = torch.nextafter(flat[0], torch.zeros_like(flat[0])
                                          if torch.isinf(flat[0]) else torch.full_like(flat[0], float("inf")))
            else:
                flat[0] = flat[0] + 1
        torch.save(st, changed)
        refuse(field, (shards[0], changed))
        if field == "batches_done":
            partial = root / "allowed-batch-count"
            _success(("scripts/merge_atlas.py", "--out", partial, "--device", "cpu",
                      "--allow-partial", shards[0], changed))
            assert (partial / "atlas.pt").is_file()
        else:
            refuse(f"{field}-partial", (shards[0], changed), allow_partial=True)


if __name__ == "__main__":
    sys.exit(0 if pytest.main([__file__, *sys.argv[1:]]) == 0 else 1)
