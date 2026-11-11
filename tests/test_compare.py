"""Blind comparison checks from specs/compare.md; no model or production data.

Run with pytest or ``python -m tests.test_compare`` (exit status 0/1).
Implementation imports are deferred so collection works before unit 6 lands.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

from atlas.ruler import AA, DESC, LEVELS, SEQ_STRIDE, Tables, calls, score_column

ROOT = Path(__file__).absolute().parents[1]


@pytest.fixture
def compare():
    from atlas.compare import column_stats, share_ci, summarise

    return column_stats, share_ci, summarise


@pytest.fixture(scope="module")
def synthetic():
    proteins = []
    pos = torch.arange(60)
    for i in range(48):
        ss3 = (pos // len(AA) + i) % 3
        proteins.append({"seq_idx": i, "seq": AA * 3, "ss3": ss3,
                         "plddt": torch.full((60,), 90.0), "burial": (pos % 46).float(),
                         "phi": torch.full((60,), -60.0),
                         "psi": torch.where(ss3 == 1, -40.0, 120.0), "cluster": str(i)})
    tables = Tables(proteins)
    gen = torch.Generator().manual_seed(27)
    tokens = {"random": [], "helix": [], "lys_helix": []}
    for p in proteins:
        helix = (p["ss3"] == 1).nonzero().flatten()
        lys = next(j for j in helix.tolist() if p["seq"][j] == "K")
        base = p["seq_idx"] * SEQ_STRIDE
        tokens["random"].append(base + int(torch.randint(60, (), generator=gen)))
        tokens["helix"].append(base + int(helix[p["seq_idx"] % len(AA)]))
        tokens["lys_helix"].append(base + lys)
    reservoirs = {name: torch.tensor(tok).repeat(max(LEVELS.values()) + 1, 1)
                  for name, tok in tokens.items()}
    scores = {name: score_column(tables, tok, B=200, seed=i)
              for i, (name, tok) in enumerate(reservoirs.items())}
    return tables, reservoirs, scores


def _score(p=0.002, n_desc=3):
    """Hand-built score_column structure for exact p and tie boundaries."""
    score = {level: {"n_samples": 20, "n_clusters": 20} for level in LEVELS}
    peak = score["peak"]
    peak["aa"] = {"n": 20, "p": 0.5, "eff_residues": 20.0,
                  "top": [(r, 0.05) for r in AA[:4]],
                  "p_res": [1.0] * 20, "log2_res": [0.0] * 20}
    peak["aa_given_fine"] = {}
    for k, size in list(DESC.items())[:n_desc]:
        peak[k] = {"n": 20, "p": p, "obs": [20.0] + [0.0] * (size - 1),
                   "exp": [20.0 / size] * size, "log2": [0.0] * size,
                   "p_state": [1.0] * size, "ci_lo": [0.0] * size, "ci_hi": [0.0] * size}
        peak["aa_given_fine"][k] = {"n": 20, "p": 0.75, "p_max": 0.6,
                                    "p_res": [1.0] * 20, "log2_res": [0.0] * 20}
    return score


def _oracle(column_stats, scores):
    original = copy.deepcopy(scores)
    oracle_scores = copy.deepcopy(scores)
    oracle_calls = calls(oracle_scores, rule="v1.2")
    for score, annotated, call in zip(scores, oracle_scores, oracle_calls):
        record = column_stats(copy.deepcopy(score))
        insufficient = call["call"] == "insufficient samples"
        assert record["insufficient"] is insufficient
        assert record["n_clusters_peak"] == score["peak"]["n_clusters"]
        fields = ("primary", "n_desc", "p_struct", "p_struct_corr", "primary_jsd", "p_res", "residue_modes")
        if insufficient:
            assert all(record[key] is None for key in fields)
            continue
        primary = annotated["primary"]
        peak = score["peak"]
        n_desc = sum(bool(peak.get(k)) for k in DESC)
        assert record["primary"] == primary
        assert record["primary_jsd"] == annotated["primary_jsd"]
        assert record["n_desc"] == n_desc
        assert record["p_struct"] == peak[primary]["p"]
        assert record["p_struct_corr"] == min(1.0, peak[primary]["p"] * n_desc)
        residue = peak.get("aa_given_fine", {}).get(primary)
        assert record["p_res"] == (residue["p_max"] if residue else None)
        assert record["residue_modes"] == call["residue_modes"]
    assert scores == original  # calls() only received the deep copy.


def test_column_stats_real_scores_match_calls(compare, synthetic):
    column_stats, _, _ = compare
    _oracle(column_stats, list(synthetic[2].values()))


@pytest.mark.parametrize("n_desc", [1, 2, 3])
@pytest.mark.parametrize("p", [0.002, 0.01, 0.7])
def test_structural_correction_and_clipping(compare, n_desc, p):
    _oracle(compare[0], [_score(p, n_desc)])


def test_primary_p_then_jsd_and_residue_boundaries(compare):
    column_stats, _, _ = compare
    tied = _score()
    # Equal p: a concentrated distribution beats obs == exp (zero JSD).
    tied["peak"]["state5"]["obs"] = tied["peak"]["state5"]["exp"][:]
    tied["peak"]["rama4"]["p"] = 0.5
    assert column_stats(copy.deepcopy(tied))["primary"] == "burial3"
    lower_p = copy.deepcopy(tied)
    lower_p["peak"]["state5"]["p"] = 0.001
    assert column_stats(copy.deepcopy(lower_p))["primary"] == "state5"
    residue = tied["peak"]["aa_given_fine"]["burial3"]
    residue["p_res"][:4] = [0.009, 0.01, 0.001, 0.009]
    residue["log2_res"][:4] = [1.0, 2.0, 0.999, 2.0]
    assert column_stats(copy.deepcopy(tied))["residue_modes"] == [AA[0], AA[3]]
    _oracle(column_stats, [tied, lower_p])


def test_missing_residue_test_and_insufficient(compare, synthetic):
    column_stats, _, _ = compare
    no_residue = _score()
    del no_residue["peak"]["aa_given_fine"]
    no_descriptors = _score(n_desc=0)
    tables, reservoirs, _ = synthetic
    few = score_column(tables, reservoirs["random"][:, :9], B=200, seed=4)
    ten = score_column(tables, reservoirs["random"][:, :10], B=200, seed=4)
    assert column_stats(few)["insufficient"] is True
    assert column_stats(ten)["insufficient"] is False
    _oracle(column_stats, [no_residue, no_descriptors, few, ten])


def _record(column_stats, score, unit=0, source="trained"):
    return dict(column_stats(copy.deepcopy(score)), source=source, layer=0, unit=unit,
                col=unit, seed=unit, pole=0,
                score_sha256=hashlib.sha256(json.dumps(score, sort_keys=True).encode()).hexdigest())


def test_records_invariant_to_scoring_and_pooling(compare, synthetic):
    column_stats, _, summarise = compare
    tables, reservoirs, scores = synthetic
    trained = _record(column_stats, scores["random"])
    others = [_record(column_stats, score, i + 1, "shuffled")
              for i, score in enumerate(scores.values())]
    for repetitions in (0, 1, 3):
        rescored = score_column(tables, reservoirs["random"], B=200, seed=0)
        record = _record(column_stats, rescored)
        assert record == trained
        pool = [record, *copy.deepcopy(others * repetitions)]
        before = copy.deepcopy(pool)
        summarise(pool)
        assert pool == before
        assert pool[0] == trained
        # Repeating another source must not affect this source's primary statistics.
        assert summarise([r for r in pool if r["source"] == "trained"]) == summarise([trained])


def _beta_quantile(prob, a, b):
    """I_x(a,b) = P[Binomial(a+b-1,x) >= a], inverted independently."""
    n = a + b - 1
    lo, hi = 0.0, 1.0
    for _ in range(90):
        x = (lo + hi) / 2
        cdf = math.fsum(math.comb(n, j) * x ** j * (1 - x) ** (n - j)
                        for j in range(a, n + 1))
        if cdf < prob:
            lo = x
        else:
            hi = x
    return (lo + hi) / 2


def _cp(k, n, level=0.95):
    tail = (1 - level) / 2
    return (0.0 if k == 0 else _beta_quantile(tail, k, n - k + 1),
            1.0 if k == n else _beta_quantile(1 - tail, k + 1, n - k))


@pytest.mark.parametrize("k,n", [(0, 1), (1, 1), (0, 10), (10, 10), (1, 10), (2, 3), (5, 10), (19, 20)])
@pytest.mark.parametrize("level", [0.95, 0.9, 0.99])
def test_share_ci_exact_binomial(compare, k, n, level):
    _, share_ci, _ = compare
    lo, hi = share_ci(k, n, level=level)
    assert (lo, hi) == pytest.approx(_cp(k, n, level), abs=1e-10)
    assert 0 <= lo <= k / n <= hi <= 1
    if level == 0.95:
        assert share_ci(k, n) == pytest.approx((lo, hi), abs=1e-12)
    if k == 0:
        assert lo == 0
        assert hi == pytest.approx(1 - ((1 - level) / 2) ** (1 / n))
    if k == n:
        assert hi == 1
        assert lo == pytest.approx(((1 - level) / 2) ** (1 / n))


def _worked_records():
    records = []
    for unit, (p, nd, jsd) in enumerate([(0.0005, 3, 0.12), (0.004, 3, 0.05), (0.002, 2, 0.08)]):
        records.append(dict(source="trained", layer=0, unit=unit, col=10 + unit, seed=10 + unit,
                            pole=0, insufficient=False, n_clusters_peak=20, primary="state5",
                            n_desc=nd, p_struct=p, p_struct_corr=p * nd, primary_jsd=jsd,
                            p_res=0.5, residue_modes=["K"] if unit == 1 else [], score_sha256="0" * 64))
    records.append(dict(source="trained", layer=0, unit=3, col=13, seed=13, pole=0,
                        insufficient=True, n_clusters_peak=9, score_sha256="0" * 64,
                        **{k: None for k in ("primary", "n_desc", "p_struct", "p_struct_corr",
                                            "primary_jsd", "p_res", "residue_modes")}))
    return records


def test_summarise_worked_example(compare):
    _, share_ci, summarise = compare
    result = summarise(_worked_records())
    assert (result["n_sampled"], result["n_scored"], result["k_struct"]) == (4, 3, 2)
    assert result["share_struct"] == pytest.approx(2 / 3)
    assert result["share_struct_all"] == 0.5
    assert result["share_struct_ci"] == pytest.approx(_cp(2, 3), abs=1e-10)
    assert [round(v, 4) for v in share_ci(2, 3)] == [0.0943, 0.9916]
    jsd = result["jsd"]
    assert jsd["n"] == 3
    assert jsd["mean"] == pytest.approx((0.12 + 0.05 + 0.08) / 3)
    for key, value in zip(("q10", "q25", "q50", "q75", "q90"), (0.056, 0.065, 0.08, 0.10, 0.112)):
        assert jsd[key] == pytest.approx(value)
    # Like share_struct, interpret this share over non-insufficient columns.
    assert result["share_residue_mode"] == pytest.approx(1 / 3)


def test_summary_structural_threshold_inclusive(compare):
    records = _worked_records()[:2]
    records[0].update(p_struct=0.005, n_desc=2, p_struct_corr=0.01)
    records[1].update(p_struct=0.00500001, n_desc=2, p_struct_corr=0.01000002)
    result = compare[2](records)
    assert result["k_struct"] == 1
    assert result["share_struct"] == 0.5
    assert result["share_struct_all"] == 0.5


def _cli(*args):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1")
    proc = subprocess.run([sys.executable, "scripts/compare_sources.py", *map(str, args)],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)


def test_sampling_counts_permutation_repeat_and_source_independence(tmp_path):
    seed, k = 13, 4
    paths = []
    for name, L, D in (("trained", 3, 17), ("shuffled", 2, 23)):
        path = tmp_path / f"{name}.pt"
        torch.save({"res_tok": torch.full((L, 2, D, 6, 2), -1, dtype=torch.long),
                    "top_val": torch.zeros(L, 2, D, 2)}, path)
        paths.append((name, path, L, D))

    def sample(tag, sources, sample_seed=seed, per_layer=k):
        out = tmp_path / f"{tag}.json"
        _cli("sample", "--atlas", *[f"{name}={path}" for name, path, _, _ in sources],
             "--per-layer", per_layer, "--seed", sample_seed, "--out", out)
        result = json.loads(out.read_text())
        assert result["seed"] == sample_seed
        assert result["per_layer"] == per_layer
        for name, path, L, D in sources:
            source = result["sources"][name]
            assert (source["L"], source["D"]) == (L, D)
            assert source["atlas_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            columns = [tuple(c) for c in source["columns"]]
            assert len(columns) == len(set(columns)) == L * per_layer
            assert Counter(layer for layer, _ in columns) == {layer: per_layer for layer in range(L)}
            for layer in range(L):
                expected = torch.randperm(D, generator=torch.Generator().manual_seed(sample_seed * 7919 + layer))[:per_layer]
                assert {unit for l, unit in columns if l == layer} == set(expected.tolist())
        return result["sources"]

    together = sample("together", paths)
    assert sample("repeat", paths) == together
    assert sample("reversed", paths[::-1]) == together
    for source in paths:
        assert sample(source[0], [source])[source[0]] == together[source[0]]
    sample("other-seed", paths, sample_seed=seed + 1)
    sample("all", paths, per_layer=17)


def test_cli_summary_pools_calls_once_and_preserves_primary_stats(compare, tmp_path):
    column_stats, _, summarise = compare
    # The trained call passes only when enough small p-values join its BH pool.
    scores = {"trained": [_score(p=0.006, n_desc=1)],
              "shuffled": [_score(p=0.0001, n_desc=1) for _ in range(3)]}
    for score in scores["shuffled"]:
        score["peak"]["aa_given_fine"]["state5"]["p_max"] = 0.0001
    records = {}
    ruler_sha256 = hashlib.sha256((ROOT / "atlas/ruler.py").read_bytes()).hexdigest()
    for name, full_scores in scores.items():
        records[name] = [_record(column_stats, s, i, name) for i, s in enumerate(full_scores)]
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records[name]))
        torch.save(full_scores, tmp_path / f"{name}.scores.pt")
        (tmp_path / f"{name}.meta.json").write_text(json.dumps(
            {"B": 200, "tables_sha256": "1" * 64, "ruler_sha256": ruler_sha256, "seconds": 0.0, "workers": 1}))
    trained_calls = []
    for pool in (["trained"], ["trained", "shuffled"]):
        out = tmp_path / ("summary-" + "-".join(pool) + ".json")
        _cli("summarise", "--scores", tmp_path, "--pool", ",".join(pool), "--out", out)
        result = json.loads(out.read_text())
        assert result["tables_sha256"] == "1" * 64
        assert result["ruler_sha256"] == ruler_sha256
        assert result["B"] == 200
        for name in pool:
            expected = summarise(copy.deepcopy(records[name]))
            for key in ("n_sampled", "n_scored", "k_struct", "share_struct", "share_struct_ci",
                        "share_struct_all", "jsd", "share_residue_mode"):
                assert result["sources"][name][key] == expected[key]
        pooled = result["pooled_v12"]
        assert pooled["pool"] == pool
        assert pooled["n_columns"] == sum(len(scores[name]) for name in pool)
        oracle = calls(copy.deepcopy([s for name in pool for s in scores[name]]), rule="v1.2")
        offset = 0
        for name in pool:
            expected = Counter(c["call"] for c in oracle[offset:offset + len(scores[name])])
            actual = {call: count for call, count in pooled["calls"][name].items() if count}
            assert actual == dict(expected)
            offset += len(scores[name])
        trained_calls.append(pooled["calls"]["trained"])
    assert trained_calls[0] != trained_calls[1]


def test_ruler_byte_identity_at_spec_commit():
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1")
    commit = subprocess.run(["git", "log", "-n1", "--format=%H", "--", "specs/compare.md"],
                            cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert commit
    blob = subprocess.run(["git", "show", f"{commit}:atlas/ruler.py"], cwd=ROOT, env=env,
                          capture_output=True, check=True).stdout
    assert hashlib.sha256((ROOT / "atlas/ruler.py").read_bytes()).digest() == hashlib.sha256(blob).digest()


def main():
    return 0 if pytest.main([str(Path(__file__)), "-q", *sys.argv[1:]]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
