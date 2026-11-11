"""Unit 4 acceptance M–P from specs/scale.md.

Run with pytest or ``python -m tests.test_shard_scale``. The real SMOKE job
runs once per module; report/prediction oracles use only hand-built records.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[1]
_TMP = tempfile.TemporaryDirectory(prefix="proteus-shard-scale-", dir=ROOT)
WORK = Path(_TMP.name)
_MISSING = object()


def _env(**extra):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    env.update(extra)
    return env


def _run(args, *, env=None, timeout=60, expected=0):
    p = subprocess.run([str(x) for x in args], cwd=ROOT, env=env or _env(),
                       capture_output=True, text=True, timeout=timeout)
    if expected is not None:
        assert p.returncode == expected, (args, p.returncode, p.stdout, p.stderr)
    return p


def _json(path):
    return json.loads(path.read_text())


def _write(path, value, *, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in value) if jsonl
                    else json.dumps(value))


def _named(value, key):
    # JSON nesting is not frozen. Keep assertions on the spec's named quantities.
    if isinstance(value, dict):
        if key in value:
            yield value[key]
        for child in value.values():
            yield from _named(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from _named(child, key)


def _n_record(value, n):
    if isinstance(value, dict):
        for key in (str(n), f"n{n}"):
            if key in value:
                return value[key]
        if value.get("N", value.get("n")) == n:
            return value
        children = value.values()
    elif isinstance(value, list):
        children = value
    else:
        return _MISSING
    for child in children:
        found = _n_record(child, n)
        if found is not _MISSING:
            return found
    return _MISSING


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _metric(record, key):
    values = list(_named(record, key))
    assert len(values) == 1, (key, values, record)
    return values[0]


def _runs(value):
    if isinstance(value, dict):
        seconds = value.get("seconds")
        if isinstance(seconds, dict) and "stream_this_process" in seconds:
            yield value
        else:
            for child in value.values():
                yield from _runs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _runs(child)


def _snapshot(excluded=()):
    # Metadata only; never descend into results, git internals or input symlinks.
    # Directory mtimes also catch creation/deletion of transient child files.
    found = {}
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        base = Path(base)
        dirs[:] = [d for d in dirs if d not in ("results", ".git", ".agents", ".codex")
                   and base / d not in excluded and not (base / d).is_symlink()]
        for path in [base, *(base / name for name in files)]:
            if path in excluded:
                continue
            st = path.lstat()
            found[str(path.relative_to(ROOT))] = (st.st_mode, st.st_size, st.st_mtime_ns)
    return found


@lru_cache(maxsize=None)
def _smoke():
    data = Path(os.environ.get("PROTEUS_TEST_DATA", "data/swissprot_test500.pt"))
    if not data.is_absolute():
        data = ROOT / data
    if not data.is_file():
        pytest.skip(f"ESM-2 scale needs dataset: {data} (set PROTEUS_TEST_DATA)")
    out, scratch, fs = WORK / "smoke", WORK / "scratch", WORK / "fs"
    for path in (out, scratch, fs / "cache", WORK / "home", WORK / "tmp"):
        path.mkdir(parents=True)
    (fs / "cache" / "swissprot_2026_03.pt").symlink_to(data)
    # Keep model cache as read-only input even when HOME is redirected to a canary.
    hf_home = os.environ.get("HF_HOME", str(Path(os.environ.get(
        "XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface"))
    env = _env(SMOKE="1", JOB="test-shard-scale", OUT=str(out), SCRATCH=str(scratch),
               FS=str(fs), DEADLINE=str(int(time.time()) + 710), NS="1 2",
               PYTHON=sys.executable, SMOKE_DATA=str(data), PROTEUS_TEST_DATA=str(data),
               MODEL="8M", TPB="4096", PT="4096", K="16", KR="8",
               MAX_SEQS="32", MAX_BATCHES="4", NAIVE_PREFIX="2",
               NAIVE_STEADY="2", NAIVE_OFFSET="0", KEEP="0",
               HOME=str(WORK / "home"), TMPDIR=str(WORK / "tmp"), HF_HOME=hf_home,
               XDG_CACHE_HOME=str(WORK / "home" / ".cache"),
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    before = _snapshot((out, scratch))
    started = time.monotonic()
    p = _run(["bash", "jobs/shard_scale/run.sh"], env=env, timeout=720, expected=None)
    elapsed = time.monotonic() - started
    after = _snapshot((out, scratch))
    return out, scratch, env, p, elapsed, before, after


def test_m_smoke_scaling_metrics_and_gates():
    out, _, _, p, elapsed, _, _ = _smoke()
    assert p.returncode == 0, (p.returncode, p.stdout, p.stderr)
    assert elapsed < 720, elapsed
    report = _json(out / "scaling.json")
    for n in (1, 2):
        record = _n_record(report, n)
        assert record is not _MISSING and record is not None, n
        for key in ("speedup_e2e", "speedup_stream", "candidates_ratio"):
            assert _number(_metric(record, key)), (n, key, record)
    # N=1 is unsharded (step 5), so only N=2 has a merge process/RSS.
    merged = _metric(_n_record(report, 2), "merge")
    assert _metric(merged, "peak_rss_bytes") > 0
    runs = list(_runs(report))
    assert len(runs) >= 3, "need per-run records for single and both N=2 shards"
    for run in runs:
        for key in ("build_s", "wait_s"):
            value = _metric(run, key)
            assert _number(value) and value >= 0, (key, value)
    assert _json(out / "n2" / "gate_compare.json")["ok"] is True
    assert _metric(_metric(report, "naive"), "cmp_prefix")["ok"] is True


def test_n_report_known_arithmetic_and_missing_n4():
    out = WORK / "known-report"
    # Step-6 layout: n2/s<i>/{run.json,batches.jsonl}, n2/merged/merge.json.
    # N=1 uses single/; timeline follows shard_gate's {job, steps} envelope.
    for directory, stream, compacted, shard in (
            ("single", 48, 100, None), ("n2/s0", 12, 60, [0, 2]),
            ("n2/s1", 24, 90, [1, 2])):
        count = 4 if shard is None else 2
        _write(out / directory / "run.json", {
            "seconds": {"load": 2, "pilot": 0, "stream_this_process": stream,
                        "final_save": 1, "total": stream + 3},
            "stats": {"compacted": compacted, "top_cand": compacted * 2,
                      "res_cand": compacted * 3, "batches": count},
            "shard": shard, "n_seen": 100 * count, "n_batches_plan": 4,
            "host": {"cpus": 4, "peak_rss_bytes": 8192}})
        _write(out / directory / "batches.jsonl", [
            {"batch": b if shard is None else 2 * b + shard[0],
             "shard": shard, "shard_pos": b, "n_seen_before": 100 * b,
             "n_valid": 100, "t_start": 100 + b, "t_end": 101 + b,
             "atlas_pass_s": 3 + b, "build_s": 1 + b, "wait_s": b,
             "seconds": 5 + b, "redo_layers": 0, "compacted_frac": 0.5}
            for b in range(count)], jsonl=True)
    _write(out / "n2/merged/merge.json", {
        "validate_s": 1, "steps": [{"load_s": 1, "merge_s": 2}],
        "result_s": 1, "save_s": 1, "total_s": 6, "peak_rss_bytes": 16384,
        "shards": [], "atlas_pt_bytes": 64})
    gate = {"ok": True, "exact_all": True, "close_all": True,
            "scalars_all": True, "fields": {"gram": {"max_rel_diff": 0, "ok": True}}}
    for filename in ("n2/gate_compare.json", "n2/vs_ref.json", "single_vs_ref.json"):
        _write(out / filename, gate)
    steps = [("pilot", 8), ("single", 56), ("shards_n2", 24), ("merge_n2", 8),
             ("compare_n2", 1000), ("vs_ref_n2", 1000)]
    _write(out / "timeline.json", {"job": "known", "steps": [
        {"step": name, "start": 10000 * i, "end": 10000 * i + duration, "rc": 0}
        for i, (name, duration) in enumerate(steps)]})
    assert not (out / "n4").exists()
    _run([sys.executable, "scripts/scaling_report.py", out])
    report = _json(out / "scaling.json")
    one, two = _n_record(report, 1), _n_record(report, 2)
    # wall_1=8+56=64; wall_2=8+24+8=40 (not merge.json's 6).
    # Compare times excluded. Speedups: 64/40, 48/max(12,24).
    # Candidate ratio=(60+90)/100; imbalance=24/((12+24)/2).
    assert _metric(one, "wall_1") == 64
    assert _metric(two, "wall_N") == 40
    assert _metric(two, "speedup_e2e") == 64 / 40
    assert _metric(two, "speedup_stream") == 48 / 24
    assert _metric(two, "candidates_ratio") == (60 + 90) / 100
    assert _metric(two, "imbalance") == 24 / ((12 + 24) / 2)
    four = _n_record(report, 4)
    assert four is not _MISSING, "missing n4 inputs must yield null entries"
    if four is not None:
        for key in ("wall_N", "speedup_e2e", "speedup_stream", "candidates_ratio", "imbalance"):
            assert _metric(four, key) is None, key
    md = (out / "scaling.md").read_text()
    assert "N versus 1" in md and "filtered versus naive" in md


def _numeric_groups(value):
    # Per-shard sum key is unspecified; accept either sequence or i -> sum map.
    if isinstance(value, (dict, list)):
        children = list(value.values()) if isinstance(value, dict) else value
        if children and all(_number(v) for v in children):
            yield children
        for child in children:
            yield from _numeric_groups(child)


def test_o_prediction_hand_computed_sixteen_batches():
    directory = WORK / "known-prediction"
    batches, run, output = (directory / name for name in ("batches.jsonl", "run.json", "P.json"))
    # atlas_pass_s = 1..16; all compacted fractions constant, avoiding any
    # dependence on the unspecified smoothing/interpolation boundary policy.
    _write(batches, [{"batch": b, "atlas_pass_s": b + 1, "seconds": 2 * (b + 1),
                      "n_seen_before": 100 * b, "n_valid": 100, "compacted_frac": 0.5}
                     for b in range(16)], jsonl=True)
    _write(run, {"seconds": {"pilot": 7, "load": 999, "stream_this_process": 999,
                             "final_save": 999, "total": 999},
                 "n_seen": 1600, "stats": {"compacted": 800}, "n_batches_plan": 16})
    fixed = {"load_s": 2, "shard_save_s": 3, "merge_load_s": 5,
             "merge_step_s": 7, "result_s": 11, "save_s": 13}
    _run([sys.executable, "scripts/predict_scaling.py", "--batches", batches, "--run", run,
          "--ns", "1", "2", "4", "--fixed", json.dumps(fixed), "--out", output])
    report = _json(output)
    # N=1: sum(1..16)=136. N=2: odds=64, evens=72.
    # N=4: (1+5+9+13, 2+6+10+14, 3+7+11+15, 4+8+12+16).
    sums = {1: [136], 2: [64, 72], 4: [28, 32, 36, 40]}
    # wall_1=7+2+136+13=158; wall_2=7+2+72+3+2*(5+7)+11+13=132;
    # wall_4=7+2+40+3+4*(5+7)+11+13=124.
    walls = {1: 158, 2: 132, 4: 124}
    for n in (1, 2, 4):
        record = _n_record(report, n)
        assert record is not _MISSING and record is not None, n
        assert sums[n] in list(_numeric_groups(record)), (n, record)
        assert _metric(record, "stream_pred") == max(sums[n])
        assert _metric(record, "wall_pred") == walls[n]
        assert _metric(record, "speedup_stream_pred") == 136 / max(sums[n])
        assert _metric(record, "speedup_e2e_pred") == 158 / walls[n]


def test_p_smoke_write_boundaries_and_nonempty_out_refusal():
    out, scratch, env, p, _, before, after = _smoke()
    assert p.returncode == 0, (p.returncode, p.stdout, p.stderr)
    assert before == after, "SMOKE modified worktree/canaries outside OUT and SCRATCH"
    # Fresh refusal case: do not rerun expensive model work on successful OUT.
    occupied = WORK / "occupied"
    occupied.mkdir()
    sentinel = occupied / "keep.txt"
    sentinel.write_text("must survive refusal\n")
    refuse_env = dict(env, OUT=str(occupied), DEADLINE=str(int(time.time()) + 30))
    before = _snapshot((out, scratch))
    refused = _run(["bash", "jobs/shard_scale/run.sh"], env=refuse_env,
                   expected=None, timeout=30)
    assert refused.returncode != 0, (refused.stdout, refused.stderr)
    assert "OUT" in refused.stdout + refused.stderr, (refused.stdout, refused.stderr)
    assert list(occupied.iterdir()) == [sentinel]
    assert sentinel.read_text() == "must survive refusal\n"
    assert before == _snapshot((out, scratch)), "refused job modified files"


def main():
    return 0 if pytest.main([str(Path(__file__)), "-q", *sys.argv[1:]]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
