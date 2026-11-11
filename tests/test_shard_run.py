"""Unit 1 of specs/shard.md; also runnable with python -m tests.test_shard_run.

The legacy driver below freezes the pre-shard run_atlas control flow, without
calling run_atlas to produce its own reference. It intentionally uses the existing
Atlas algorithms: this checks driver compatibility, not historical core binaries.
"""
from __future__ import annotations

import atexit
import builtins
import copy
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1")

import pytest
import torch

from atlas.core import Atlas, AtlasConfig
from atlas.data import SwissProt, shuffled_order

ROOT = Path(__file__).resolve().parents[1]
EXACT = ("top_val", "top_tok", "res_tok", "res_val", "res_key", "sub_bin", "enr", "ann_tot", "edges")
SUMS = ("sub_sum", "sub_sq", "gram")
SCALARS = ("n_seen", "n_sub", "gram_rows")
OPTIONS = {
    "model": "8M", "control": "none", "control_seed": 17, "dtype": "fp32",
    "tokens_per_batch": 4096, "window": 8192, "pilot_tokens": 4096,
    "K": 16, "Kr": 8, "seed": 17, "max_seqs": 96, "max_batches": 5,
    "ckpt_every": 1, "gram_flush_every": 2,
}
_TMP = tempfile.TemporaryDirectory(prefix="proteus-shard-tests-")
atexit.register(_TMP.cleanup)
WORK = Path(_TMP.name)


def _data():
    path = Path(os.environ.get("PROTEUS_TEST_DATA", ROOT / "data/swissprot_test500.pt"))
    if not path.is_file():
        pytest.skip(f"real-model tests require PROTEUS_TEST_DATA (missing: {path})")
    return path.resolve()


def _sha_plan(plan):
    values = [len(plan)]
    for group in plan:
        values.extend([len(group), *group])
    return hashlib.sha256(struct.pack("<" + "q" * len(values), *values)).hexdigest()


def _digests(result):
    return {k: hashlib.sha256(result[k].contiguous().numpy().tobytes()).hexdigest()
            for k in EXACT if k in result}


def _load(path):
    assert path.is_file(), path
    return torch.load(path, map_location="cpu", weights_only=False)


def _json(path):
    assert path.is_file(), path
    return json.loads(path.read_text())


def _records(out):
    path = out / "batches.jsonl"
    assert path.is_file(), path
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _equal(x, y, name):
    if isinstance(x, torch.Tensor):
        assert x.dtype == y.dtype and x.shape == y.shape, name
        if x.is_floating_point():
            assert torch.equal(torch.isnan(x), torch.isnan(y)), name
            mask = ~torch.isnan(x)
            assert torch.equal(x[mask], y[mask]), name
        else:
            assert torch.equal(x, y), name
    else:
        assert x == y, name


def _compare(a, b, exact_sums=True):
    for key in (*EXACT, *SCALARS):
        assert key in a and key in b, key
        _equal(a[key], b[key], key)
    for key in SUMS:
        assert key in a and key in b, key
        x, y = a[key].double(), b[key].double()
        assert a[key].dtype == b[key].dtype and x.shape == y.shape, key
        rel = float((x - y).abs().max() / y.abs().max().clamp(min=1e-300))
        assert rel <= (0.0 if exact_sums else 1e-9), (key, rel)
        if exact_sums:
            _equal(a[key], b[key], key)


@lru_cache(None)
def _plan():
    ds = SwissProt(str(_data()))
    order = shuffled_order(len(ds), OPTIONS["seed"])[:OPTIONS["max_seqs"]]
    plan = ds.plan(order, OPTIONS["tokens_per_batch"], OPTIONS["window"], OPTIONS["seed"])
    plan = plan[:OPTIONS["max_batches"]]
    n_total = sum(min(int(ds.len[i]), ds.max_res) for group in plan for i in group)
    return plan, n_total


@lru_cache(None)
def _run(extra=(), directory=""):
    data = _data()
    name = directory or hashlib.sha256(repr(extra).encode()).hexdigest()[:16]
    out = WORK / name
    command = [sys.executable, "scripts/run_atlas.py", "--data", str(data),
               "--out", str(out), "--device", "cpu", "--seed-thresholds"]
    for key, value in OPTIONS.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    proc = subprocess.run(command + list(extra), cwd=ROOT, env=os.environ.copy(),
                          capture_output=True, text=True, timeout=180)
    return out, proc


def _success(extra=(), directory=""):
    out, proc = _run(tuple(extra), directory)
    assert proc.returncode == 0, (proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:])
    return out


@lru_cache(None)
def _pilot():
    path = WORK / "pilot.pt"
    out = _success(("--pilot-out", str(path), "--pilot-only"))
    assert not (out / "atlas.pt").exists()
    assert not (out / "shard.pt").exists()
    assert not (out / "batches.jsonl").exists() or not _records(out)
    artifact = _load(path)
    plan, n_total = _plan()
    assert artifact["plan_sha256"] == _sha_plan(plan)
    assert artifact["n_batches"] == len(plan)
    assert artifact["n_total"] == n_total
    expected_key = {k: OPTIONS[k] for k in (
        "model", "control", "control_seed", "dtype", "tokens_per_batch", "window",
        "pilot_tokens", "K", "Kr", "seed", "max_seqs", "max_batches")}
    expected_key.update(sorted_batches=False, edge_q=[0.5, 0.9, 0.99, 0.999, 0.9999], seed_thresholds=True)
    assert artifact["args_key"] == expected_key
    edges = artifact["edges"]
    assert edges.dtype == torch.float32 and tuple(edges.shape) == (6, 2, 1280, 5)
    assert artifact["top_seed"].dtype == torch.float32
    assert artifact["top_seed"].shape == edges.shape[:-1]
    assert artifact["res_seed_key"].dtype == torch.int64
    assert artifact["res_seed_key"].shape == edges.shape
    assert artifact["model"]["n_layers"] == 6 and artifact["model"]["d_ff"] == 1280
    return path


def _shard(i, n):
    return _success(("--pilot-from", str(_pilot()), "--shard", f"{i}/{n}"))


def _instrumentation(out, shard):
    records = _records(out)
    for record in records:
        assert record["shard"] == shard
        position = record["batch"] if shard is None else record["batch"] // shard[1]
        assert record["shard_pos"] == position
        for key in ("build_s", "wait_s"):
            assert isinstance(record[key], (int, float)) and record[key] >= 0
        assert record["gram_flushed"] == (position % OPTIONS["gram_flush_every"] == 1)
    summary = _json(out / "run.json")
    for key in ("final_save", "total"):
        assert summary["seconds"][key] >= 0
    assert summary["seconds"]["total"] >= summary["seconds"]["final_save"]
    assert summary["host"]["cpus"] >= 1
    assert summary["host"]["peak_rss_bytes"] > 0
    return records, summary


@torch.no_grad()
def _write_legacy(out):
    """Pre-shard CLI flow using public APIs; run in an isolated one-thread process."""
    from atlas.esm import attach, load, pilot
    from atlas.reference import pilot_edges, pilot_seeds

    ds = SwissProt(str(_data()))
    order = shuffled_order(len(ds), OPTIONS["seed"])[:OPTIONS["max_seqs"]]
    model, tokenizer, _ = load("8M", "none", dtype=torch.float32, device="cpu", seed=17)
    vocab = tokenizer.get_vocab()
    plan = ds.plan(order, 4096, 8192, 17)[:5]
    n_total = sum(min(int(ds.len[i]), ds.max_res) for group in plan for i in group)
    pplan = ds.pilot_plan(plan, 4096, 4096, 17)
    hp, _ = pilot(model, ds.batches(vocab, order, 4096, plan=pplan), "cpu", 1 << 40)
    edges = torch.cat([pilot_edges([h]) for h in hp])
    parts = [pilot_seeds([h], edges[li:li + 1], 16, 8, n_total, groups=pilot.last_tok // 1024)
             for li, h in enumerate(hp)]
    atlas = Atlas(6, 1280, edges, AtlasConfig(K=16, Kr=8, seed=17), n_annot=len(ds.vocab))
    atlas.seed(torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]))
    del hp, parts
    handles = attach(model, atlas.observe)
    try:
        for bi, batch in enumerate(ds.batches(vocab, order, 4096, plan=plan)):
            ctx = atlas.make_ctx(batch.n_rows, batch.valid_rows, batch.tok, batch.ann)
            atlas.begin_batch(ctx)
            model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
            redo = atlas.end_batch()
            if redo:
                atlas.begin_batch(ctx, redo_layers=redo)
                model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
                if atlas.end_batch():
                    raise RuntimeError("dense redo did not finish")
            if bi % 2 == 1:
                atlas.flush_gram()
    finally:
        for handle in handles:
            handle.remove()
    result = atlas.result()
    out.mkdir(parents=True)
    torch.save(result, out / "atlas.pt")
    (out / "run.json").write_text(json.dumps({"digests": _digests(result)}))


@lru_cache(None)
def _legacy():
    _data()
    out = WORK / "legacy"
    code = "from pathlib import Path; from tests.test_shard_run import _write_legacy; _write_legacy(Path(__import__('sys').argv[1]))"
    proc = subprocess.run([sys.executable, "-c", code, str(out)], cwd=ROOT,
                          env=os.environ.copy(), capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, (proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:])
    return out


def test_a_plain_run_preserves_pre_shard_outputs():
    out, legacy = _success(), _legacy()
    result = _load(out / "atlas.pt")
    _compare(result, _load(legacy / "atlas.pt"))
    summary = _json(out / "run.json")
    assert summary["digests"] == _json(legacy / "run.json")["digests"] == _digests(result)
    assert not (out / "shard.pt").exists()
    assert "shard" not in summary["args"] or summary["args"]["shard"] is None
    _instrumentation(out, None)


def test_b_shared_pilot_matches_plain_run():
    out = _success(("--pilot-from", str(_pilot())))
    result, plain = _load(out / "atlas.pt"), _load(_success() / "atlas.pt")
    _compare(result, plain)
    assert _json(out / "run.json")["digests"] == _digests(plain)
    _equal(result["edges"], _load(_pilot())["edges"], "pilot edges")


def test_c_pilot_argument_mismatches_refuse_before_streaming():
    pilot = _pilot()
    for flag, value in (("--seed", "18"), ("--K", "17"),
                        ("--max-seqs", "95"), ("--tokens-per-batch", "8192")):
        out, proc = _run(("--pilot-from", str(pilot), flag, value))
        assert proc.returncode != 0, flag
        assert not (out / "atlas.pt").exists(), flag
        assert not (out / "shard.pt").exists(), flag
        assert not (out / "ckpt").exists() or not list((out / "ckpt").glob("*.pt")), flag
        assert not (out / "batches.jsonl").exists() or not _records(out), flag


def test_d_shards_partition_global_batches():
    plan, n_total = _plan()
    assert len(plan) >= 5, "test data must yield at least five global batches"
    for n in (2, 3):
        seen = set()
        for i in range(n):
            out = _shard(i, n)
            records, summary = _instrumentation(out, [i, n])
            batches = [r["batch"] for r in records]
            expected = list(range(i, len(plan), n))
            assert batches == expected
            assert seen.isdisjoint(batches)
            seen.update(batches)
            shard = _load(out / "shard.pt")
            assert not (out / "atlas.pt").exists()
            assert shard["shard"] == summary["shard"] == [i, n]
            assert shard["plan_sha256"] == summary["plan_sha256"] == _sha_plan(plan)
            assert shard["n_batches_plan"] == len(plan)
            assert shard["n_total_plan"] == n_total
            assert shard["batches_done"] == summary["batches_done"] == len(expected)
            assert "shard" in shard["args_key"]
            for key in ("model", "coverage", "vocab", "order_seed"):
                assert key in shard, key
            state = shard["atlas"]
            assert state["n_seen"] == sum(r["n_valid"] for r in records)
            assert state["gram64"].dtype == torch.float64
            assert state["gram_rows"] == state["n_sub"]
            artifact = _load(_pilot())
            _equal(state["E"], artifact["edges"], "shared edges")
            _equal(state["top_seed"], artifact["top_seed"].reshape(-1), "global top seed")
            keys = artifact["res_seed_key"].reshape(-1)
            expected_seed = torch.where(keys >= 2**32, torch.full_like(keys, torch.iinfo(torch.int64).min), -keys)
            _equal(state["res_seed"], expected_seed, "global reservoir seed")
        assert seen == set(range(len(plan)))
    # N larger than the plan must still produce a valid empty state.
    out = _shard(len(plan), len(plan) + 1)
    empty = _load(out / "shard.pt")
    assert _records(out) == []
    assert empty["batches_done"] == empty["atlas"]["n_seen"] == empty["atlas"]["n_sub"] == 0
    assert empty["atlas"]["gram_rows"] == 0
    assert torch.count_nonzero(empty["atlas"]["gram64"]) == 0
    assert empty["n_batches_plan"] == len(plan) and empty["n_total_plan"] == n_total
    assert empty["plan_sha256"] == _sha_plan(plan)
    assert not (out / "atlas.pt").exists()


def test_e_shard_stop_resume_matches_uninterrupted_state():
    full = _load(_shard(0, 2) / "shard.pt")
    flags = ("--pilot-from", str(_pilot()), "--shard", "0/2")
    part = _success(flags + ("--stop-after", "2"), "resume")
    assert not (part / "shard.pt").exists() and not (part / "atlas.pt").exists()
    checkpoints = list((part / "ckpt").glob("ckpt_*.pt"))
    assert checkpoints
    checkpoint = _load(max(checkpoints, key=lambda p: int(p.stem.split("_")[1])))
    # First two owned batches are global 0 and 2; next owned batch is 4.
    assert 2 < checkpoint["next_batch"] <= 4
    assert "shard" in checkpoint["args_key"]
    resumed = _success(flags + ("--resume",), "resume")
    got = _load(resumed / "shard.pt")
    for key in (*Atlas._STATE, "enr", "ann_tot", "n_seen", "n_sub"):
        assert key in got["atlas"] and key in full["atlas"], key
        _equal(got["atlas"][key], full["atlas"][key], key)
    for key in ("shard", "plan_sha256", "n_batches_plan", "n_total_plan", "batches_done", "args_key"):
        assert got[key] == full[key], key
    assert got["batches_done"] == 3
    assert _json(resumed / "run.json")["batches_done"] == 3
    assert not (resumed / "atlas.pt").exists()


def test_plan_digest_little_endian_int64():
    from atlas.data import plan_digest

    plans = [[], [[]], [[0, 1], [2]], [[0], [1, 2]], [[2, 1], [0]], [[2**40, 7], []]]
    got = [plan_digest(plan) for plan in plans]
    assert got == [_sha_plan(plan) for plan in plans]
    assert len(set(got)) == len(plans)
    assert all(isinstance(digest, str) and len(digest) == 64 for digest in got)


def test_result_digests_raw_contiguous_bytes_and_present_fields():
    from atlas.core import result_digests

    result = {}
    for key in EXACT:
        dtype = torch.float32 if key in ("top_val", "res_val", "edges") else torch.int64
        result[key] = torch.arange(12, dtype=dtype).reshape(3, 4).T
    result["res_val"][0, 0] = float("nan")
    result["top_val"][1, 0] = -0.0
    result.update(sub_sum=torch.ones(2, dtype=torch.float64), n_seen=10, cfg={})
    snapshot = copy.deepcopy(result)
    assert result_digests(result) == _digests(result)
    for key in EXACT:
        _equal(result[key], snapshot[key], key)
    assert result_digests({}) == {}
    assert result_digests({"top_tok": result["top_tok"], "sub_sum": result["sub_sum"]}) == {
        "top_tok": _digests(result)["top_tok"]}


def _synthetic(gram=True):
    edges = torch.tensor([0., 0.5, 1., 2., 4.]).expand(2, 2, 3, 5).clone()
    cfg = AtlasConfig(K=3, Kr=2, rho_res=1., rho_sub=1., cap_frac=1., gram=gram, seed=7)
    atlas = Atlas(2, 3, edges, cfg, n_annot=2)
    atlas.seed(torch.full((2, 2, 3), -float("inf")), torch.full((2, 2, 3, 5), 2**32, dtype=torch.int64))
    h = torch.tensor([[1., -2., 3.], [-4., 2., 0.5], [0.25, -1., 5.], [2., 3., -0.5]])
    rows = torch.arange(4)
    ctx = atlas.make_ctx(4, rows, rows + 1024, torch.tensor([[0, -1], [1, -1], [0, 1], [-1, -1]]))
    atlas.begin_batch(ctx)
    atlas.observe(0, h)
    atlas.observe(1, h * 2)
    if atlas.end_batch():
        raise RuntimeError("small synthetic atlas unexpectedly overflowed")
    return atlas


@pytest.mark.parametrize("gram", [True, False])
def test_state_dict_include_gram_flushes_raw_host_accumulator(gram):
    atlas = _synthetic(gram)
    before = copy.deepcopy(atlas.state_dict())
    assert "gram64" not in before and "gram_rows" not in before
    if gram:
        # Deliberately asymmetric host and pending buffers catch symmetrisation.
        atlas.gram64.copy_(torch.arange(18, dtype=torch.float64).reshape(2, 3, 3))
        atlas.gram.copy_(torch.arange(18, 36, dtype=torch.float32).reshape(2, 3, 3))
        expected = atlas.gram64.clone() + atlas.gram.double()
        rows = atlas.gram_rows
        atlas.state_dict()  # The default must not flush pending Gram work.
        _equal(atlas.gram64, torch.arange(18, dtype=torch.float64).reshape(2, 3, 3), "default host")
    state = copy.deepcopy(atlas.state_dict(include_gram=True))
    for key, value in before.items():
        _equal(state[key], value, key)
    if gram:
        _equal(state["gram64"], expected, "raw Gram")
        assert state["gram64"].dtype == torch.float64 and state["gram64"].device.type == "cpu"
        assert state["gram_rows"] == rows > 0
        _equal(atlas.gram, torch.zeros_like(atlas.gram), "flushed buffer")
        _equal(atlas.state_dict(include_gram=True)["gram64"], expected, "second flush")
    else:
        assert "gram64" not in state
        assert state["gram_rows"] == 0


def test_load_state_dict_restores_gram_and_keeps_legacy_reset():
    source = _synthetic()
    state = copy.deepcopy(source.state_dict())
    state["gram64"] = torch.arange(18, dtype=torch.float64).reshape(2, 3, 3)
    state["gram_rows"] = 29
    target = _synthetic()
    target.load_state_dict(state)
    restored = target.state_dict(include_gram=True)
    for key, value in state.items():
        _equal(restored[key], value, key)
    legacy = copy.deepcopy(state)
    del legacy["gram64"], legacy["gram_rows"]
    target.load_state_dict(legacy)
    reset = target.state_dict(include_gram=True)
    assert reset["gram_rows"] == 0
    _equal(reset["gram64"], torch.zeros_like(state["gram64"]), "legacy reset")
    for key, value in legacy.items():
        _equal(reset[key], value, key)


@pytest.mark.parametrize("backend", ["torch", "triton"])
@pytest.mark.parametrize("gram", [True, False])
def test_from_state_cpu_preserves_config_and_allows_merge(monkeypatch, backend, gram):
    state = copy.deepcopy(_synthetic(gram).state_dict(include_gram=True))
    state["cfg"]["backend"] = backend
    original = copy.deepcopy(state)
    import_module = builtins.__import__

    def without_triton(name, *args, **kwargs):
        if "triton" in name:
            raise ImportError("Triton unavailable in this CPU-only test")
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_triton)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    restored = Atlas.from_state(state)  # Default device must be CPU.
    for key, value in original.items():
        _equal(state[key], value, f"input {key}")
    saved = restored.state_dict(include_gram=True)
    for key, value in original.items():
        _equal(saved[key], value, key)
    result = restored.result()
    assert result["cfg"] == original["cfg"]
    assert all(value.device.type == "cpu" for value in result.values() if isinstance(value, torch.Tensor))
    # An empty compatible state makes this a disjoint merge with a known result.
    empty = Atlas(2, 3, state["E"], AtlasConfig(**{**state["cfg"], "backend": "torch"}), n_annot=2)
    empty_state = copy.deepcopy(empty.state_dict(include_gram=True))
    empty_state["cfg"] = copy.deepcopy(state["cfg"])
    empty_state["top_seed"] = state["top_seed"].clone()
    empty_state["res_seed"] = state["res_seed"].clone()
    other = Atlas.from_state(empty_state, device="cpu")
    restored.merge_from(other)
    merged = restored.result()
    for key in (*EXACT, "sub_sum", "sub_sq", "n_seen", "n_sub"):
        _equal(merged[key], result[key], key)
    if gram:
        _equal(merged["gram"], result["gram"], "merged Gram")
        assert merged["gram_rows"] == result["gram_rows"]
    assert merged["cfg"] == original["cfg"]


if __name__ == "__main__":
    sys.exit(0 if pytest.main([__file__, *sys.argv[1:]]) == 0 else 1)
