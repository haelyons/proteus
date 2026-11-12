"""Blind acceptance tests for section 1 of frozen specs/units.md.

Only the documented tuple forms of Column and Edit are used. Implementation
imports remain inside fixtures/tests so this file collects before they exist.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).absolute().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"


def _bytes(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _same_bytes(left, right):
    assert left.shape == right.shape
    assert left.dtype == right.dtype
    assert _bytes(left) == _bytes(right)


@pytest.fixture(scope="module")
def engine():
    from atlas.intervene import Intervener

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield Intervener(model="8M", device="cpu", dtype=torch.float32)
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def tokenizer():
    from atlas.esm import MODELS
    from transformers import EsmTokenizer

    return EsmTokenizer.from_pretrained(MODELS["8M"], local_files_only=True)


@pytest.fixture(scope="module")
def specimen(engine, tokenizer):
    data = torch.load(ROOT / "data/swissprot_test500.pt", map_location="cpu",
                      weights_only=False)
    # A short, complete protein keeps each invariant cheap and exercises both ends.
    sequence = next(seq for seq in data["seq"] if 12 <= len(seq) <= 128)
    ids = tokenizer(sequence, return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        baseline = engine.run(ids, capture=True)
    return ids, baseline


def _direction(width, kind):
    if kind == "axis":
        r = torch.zeros(width, dtype=torch.float32)
        r[7] = 1
        return r
    r = torch.randn(width, generator=torch.Generator().manual_seed(19))
    return r / r.norm()


def test_run_and_head_contract(engine, tokenizer, specimen):
    ids, baseline = specimen
    n = ids.shape[1] - 2
    assert baseline.lp.shape == (n, 20)
    assert baseline.lp_full.shape[0] == n + 2
    assert baseline.lp.dtype == baseline.lp_full.dtype == torch.float64
    assert baseline.x_final.shape == (n, 320)
    assert set(baseline.post) == set(range(6))
    for post in baseline.post.values():
        assert post.shape == (n, 1280)
        assert post.dtype == torch.float32
        assert post.device.type == "cpu"
    indices = tokenizer.convert_tokens_to_ids(list(AA))
    _same_bytes(baseline.lp, baseline.lp_full[1:-1, indices])
    with torch.inference_mode():
        actual = engine.head(baseline.x_final)
    torch.testing.assert_close(actual, baseline.lp, rtol=0, atol=1e-6)


@pytest.mark.parametrize("layer", [0, 3, 5])
@pytest.mark.parametrize("kind", ["axis", "direction"])
def test_self_replacement_is_bitwise_identity(engine, specimen, layer, kind):
    ids, baseline = specimen
    r = _direction(baseline.post[layer].shape[1], kind)
    column = (layer, r)
    positions = [0, ids.shape[1] // 2 - 1, ids.shape[1] - 3]
    a = baseline.post[layer][positions] @ r
    with torch.inference_mode():
        values = engine.column_values(baseline, column)
        edited = engine.run(ids, edit=(layer, r, positions, a.tolist()), capture=True)
        result = engine.replace(ids, column, positions, a.tolist(), baseline=baseline)
    _same_bytes(values, baseline.post[layer] @ r)
    _same_bytes(edited.lp_full, baseline.lp_full)
    assert set(edited.post) == set(baseline.post)
    for later in baseline.post:
        _same_bytes(edited.post[later], baseline.post[later])
    assert torch.count_nonzero(result.dlp) == 0
    assert torch.count_nonzero(result.write) == 0


@pytest.mark.parametrize("layer", [0, 5])
def test_neuron_and_its_axis_have_bitwise_equal_results(engine, specimen, layer):
    ids, baseline = specimen
    # Both representations are (layer, r) in the frozen interface, even for neurons.
    neuron = torch.eye(1, 1280, dtype=torch.float32).flatten().roll(7)
    direction = torch.zeros(1280, dtype=torch.float32)
    direction[7] = 1
    positions = [1, ids.shape[1] - 4]
    values = (baseline.post[layer][positions, 7] + 2).tolist()
    with torch.inference_mode():
        first = engine.replace(ids, (layer, neuron), positions, values, baseline=baseline)
        second = engine.replace(ids, (layer, direction), positions, values, baseline=baseline)
    _same_bytes(first.dlp, second.dlp)
    _same_bytes(first.write, second.write)
    assert len(first.later) == len(second.later)
    for a, b in zip(first.later, second.later):
        assert a[:2] == b[:2]
        assert struct.pack("!d", float(a[2])) == struct.pack("!d", float(b[2]))
    assert first.later_share == second.later_share
    if layer == 5:
        assert first.later == []
        assert first.later_share is None


@pytest.mark.parametrize("layer", [0, 3, 5])
@pytest.mark.parametrize("kind", ["axis", "direction"])
def test_edit_rule_and_untouched_positions(engine, specimen, layer, kind):
    ids, baseline = specimen
    r = _direction(1280, kind)
    positions = [0, 2, ids.shape[1] - 3]
    original = baseline.post[layer].clone()
    a = original[positions] @ r
    values = a + torch.tensor([1.25, 0, -2.0])
    expected = original.clone()
    changed = values != a
    selected = torch.tensor(positions)[changed]
    expected[selected] += (values[changed] - a[changed])[:, None] * r[None, :]
    with torch.inference_mode():
        edited = engine.run(ids, edit=(layer, r, positions, values.tolist()), capture=True)
    outside = torch.ones(original.shape[0], dtype=torch.bool)
    outside[positions] = False
    _same_bytes(edited.post[layer][outside], original[outside])
    _same_bytes(edited.post[layer], expected)
    _same_bytes(baseline.post[layer], original)


def test_later_ranking_and_share_are_summed_absolute_changes(engine, specimen):
    ids, baseline = specimen
    layer, r, positions = 3, _direction(1280, "direction"), [2]
    value = float(baseline.post[layer][2] @ r) + 3
    with torch.inference_mode():
        edited = engine.run(ids, edit=(layer, r, positions, [value]), capture=True)
        result = engine.replace(ids, (layer, r), positions, [value], k_later=10,
                                baseline=baseline)
    changes = [(l, u, float(change)) for l in sorted(baseline.post) if l > layer
               for u, change in enumerate((edited.post[l] - baseline.post[l]).abs().sum(0))]
    ranked = sorted(changes, key=lambda row: (-row[2], row[0], row[1]))[:10]
    assert [(l, u) for l, u, _ in result.later] == [(l, u) for l, u, _ in ranked]
    assert [c for _, _, c in result.later] == pytest.approx([c for _, _, c in ranked])
    assert result.later_share == pytest.approx(sum(c for _, _, c in ranked) /
                                             sum(c for _, _, c in changes))
    _same_bytes(result.dlp, edited.lp - baseline.lp)


def _effects(ids, baseline, edited, position):
    lp0, lp1 = baseline.lp_full, edited.lp_full
    kl = (lp0.exp() * (lp0 - lp1)).sum(-1).clamp(min=0)
    other = torch.cat([kl[1:position + 1], kl[position + 2:-1]])
    token = int(ids[0, position + 1])
    return {"dlogp_true": float(lp1[position + 1, token] - lp0[position + 1, token]),
            "kl": float(kl[position + 1]),
            "travel_kl": float(other.max()) if other.numel() else 0.0}


def test_recorded_unit_writes_reproduction(request, record_property):
    path = os.environ.get("PROTEUS_SWISSPROT")
    if not path:
        pytest.skip("PROTEUS_SWISSPROT unset: full Swiss-Prot cache required")
    from atlas.intervene import Intervener

    tokenizer = request.getfixturevalue("tokenizer")
    # The recorded results directory is opened only for this explicitly enabled test.
    cases = [json.loads(line) for line in
             (ROOT / "results/unit_writes_8M_20261105/cases.jsonl").read_text().splitlines()
             if line.strip()]
    assert len(cases) == 900
    data = torch.load(path, map_location="cpu", weights_only=False)
    sequences = dict(zip(data["acc"], data["seq"]))
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    n_bitwise, failures = 0, []
    maximum = dict.fromkeys(("dlogp_true", "kl", "travel_kl"), 0.0)
    try:
        engine = Intervener(model="8M", device="cpu", dtype=torch.float32)
        # Group by accession: keep only one baseline in memory.
        for accession in dict.fromkeys(row["acc"] for row in cases):
            ids = tokenizer(sequences[accession][:1022], return_tensors="pt")["input_ids"]
            with torch.inference_mode():
                baseline = engine.run(ids)
                for row in (row for row in cases if row["acc"] == accession):
                    exact = True
                    for field, prefix in (("unit", ""), ("base_unit", "base_")):
                        r = torch.zeros(1280, dtype=torch.float32)
                        r[row[field]] = 1
                        edited = engine.run(ids, edit=(row["layer"], r, [row["pos"]], [0.0]))
                        for name, value in _effects(ids, baseline, edited, row["pos"]).items():
                            expected = row[prefix + name]
                            maximum[name] = max(maximum[name], abs(value - expected))
                            exact &= struct.pack("!d", value) == struct.pack("!d", expected)
                            rel = 0 if name == "dlogp_true" else 1e-4
                            if value != pytest.approx(expected, abs=1e-6, rel=rel):
                                failures.append((accession, row["layer"], row[field], name,
                                                 value, expected))
                    n_bitwise += int(exact)
    finally:
        torch.set_num_threads(previous)
        record_property("n_bitwise", n_bitwise)
        record_property("max_abs_diff", json.dumps(maximum))
        print(f"n_cases={len(cases)} n_bitwise={n_bitwise} max_abs_diff={maximum}")
    assert not failures, failures[:20]
