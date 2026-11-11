"""Blind invariants and arithmetic from specs/planted.md; no model or real data.

Run with pytest or ``python -m tests.test_planted`` (exit status 0/1).
Only subprocesses exercise synthetic_columns; no private implementation oracle.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).absolute().parents[1]
SPEC_COMMIT = "1acc99f72c6e6115dcb3970aced0130a5b6028b2"
AA = "ACDEFGHIKLMNPQRSTVWY"
RES_ORDER = "KEDRLAQSTNVIGFYHMWCP"
STRIDE = 1024


def _env():
    return dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1")


def _run(args, cwd=ROOT):
    result = subprocess.run(list(map(str, args)), cwd=cwd, env=_env(),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (args, result.stdout, result.stderr)
    return result


def _blob(path):
    return subprocess.run(["git", "show", f"{SPEC_COMMIT}:{path}"], cwd=ROOT,
                          env=_env(), capture_output=True, check=True).stdout


def _input(path, n=400, length=200, rotate=False):
    path.mkdir()
    gen = torch.Generator().manual_seed(519)
    residues = torch.randint(20, (n, length), generator=gen)
    states = torch.randint(3, (n, length), generator=gen)
    # Include every residue/state pair, then the spec's worked example verbatim.
    pairs = torch.randperm(60, generator=gen)
    residues[:, :60] = pairs % 20
    states[:, :60] = pairs // 20
    residues[0, :4] = torch.tensor([AA.index(c) for c in "KEAK"])
    states[0, :4] = torch.tensor([1, 1, 0, 1])
    if rotate:
        states = (states + 1) % 3
    seqs = ["".join(AA[i] for i in row) for row in residues.tolist()]
    acc = [f"FAKE{i:06d}" for i in range(n)]
    torch.save({"acc": acc, "seq": seqs, "len": torch.full((n,), length, dtype=torch.int32),
                "iv": torch.empty((0, 4), dtype=torch.int32),
                "vocab": [f"aa:{c}" for c in AA] + ["aa:other"], "release": "fake"},
               path / "swissprot.pt")
    desc = path / "desc"
    desc.mkdir()
    torch.save({"acc": acc, "seq_idx": torch.arange(n),
                "offset": torch.arange(n + 1) * length,
                "heldout": torch.zeros(n, dtype=torch.bool), "ss3": states.flatten(),
                "plddt": torch.full((n * length,), 90.0),
                "burial": torch.randint(46, (n * length,), generator=gen).float(),
                "phi": torch.full((n * length,), -60.0),
                "psi": torch.where(states.flatten() == 1, -40.0, 120.0)}, desc / "part_0.pt")
    with gzip.open(path / "uniref.tsv.gz", "wt") as stream:
        stream.write("acc\tcluster\n")
        stream.writelines(f"{a}\tUniRef50_{a}\n" for a in acc)
    return dict(path=path, seq=seqs, states=states, n=n * length, length=length)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("planted-input")
    return {"large": _input(root / "large"),
            "randdir": _input(root / "randdir", n=80),
            "small": _input(root / "small", n=4, length=64),
            "rotated": _input(root / "rotated", n=4, length=64, rotate=True)}


@pytest.fixture(scope="module")
def generate(tmp_path_factory, inputs):
    root = tmp_path_factory.mktemp("planted-output")
    cache = {}

    def run(name, source="large", flags=(), kr=2048, pilot=2000, script=None, path_only=False):
        if name not in cache:
            data = inputs[source]["path"]
            out = root / name
            _run([sys.executable, script or ROOT / "scripts/synthetic_columns.py",
                  "--data", data / "swissprot.pt", "--desc-dir", data / "desc",
                  "--uniref", data / "uniref.tsv.gz", "--out", out,
                  "--tokens-per-batch", 1024, "--pilot-tokens", pilot,
                  "--replicates", 1, "--Kr", kr, "--seed", 19, "--noise-seed", 7, *flags])
            cache[name] = out
        out = cache[name]
        if path_only:
            return out
        return (torch.load(out / "synthetic_atlas.pt", map_location="cpu", weights_only=False),
                json.loads((out / "columns.json").read_text()))

    return run


def _bits_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape
        # Includes NaN payloads and signed zeros, unlike allclose/equal_nan.
        assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                           b.contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _bits_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            _bits_equal(x, y)
    else:
        assert type(a) is type(b) and a == b


def test_default_bitwise_at_spec_commit(generate, tmp_path):
    # Execute the historical blob without inspecting it or importing its helpers.
    historical = tmp_path / "historical"
    (historical / "scripts").mkdir(parents=True)
    shutil.copytree(ROOT / "atlas", historical / "atlas", ignore=shutil.ignore_patterns("__pycache__"))
    script = historical / "scripts/synthetic_columns.py"
    script.write_bytes(_blob("scripts/synthetic_columns.py"))
    current = generate("default", "small", kr=256, pilot=256)
    reference = generate("historical", "small", kr=256, pilot=256, script=script)
    _bits_equal(current, reference)
    current_json = generate("default", path_only=True) / "columns.json"
    reference_json = generate("historical", path_only=True) / "columns.json"
    assert current_json.read_bytes() == reference_json.read_bytes()


def test_ruler_byte_identity_at_spec_commit():
    assert (ROOT / "atlas/ruler.py").read_bytes() == _blob("atlas/ruler.py")


def _stored(atlas, col, top=False):
    """Join stored values by token, across both poles; never assume bin preservation."""
    tok = atlas["res_tok"][0, :, col].flatten()
    val = atlas["res_val"][0, :, col].flatten()
    key = atlas["res_key"][0, :, col].flatten()
    if top:
        tok = torch.cat((tok, atlas["top_tok"][0, :, col].flatten()))
        val = torch.cat((val, atlas["top_val"][0, :, col].flatten()))
        key = torch.cat((key, torch.zeros_like(atlas["top_tok"][0, :, col].flatten())))
    valid = (tok >= 0) & torch.isfinite(val)
    tok, val, key = tok[valid], val[valid], key[valid]
    order = tok.argsort(stable=True)
    tok, val, key = tok[order], val[order], key[order]
    first = torch.cat((torch.ones(1, dtype=torch.bool), tok[1:] != tok[:-1]))
    return tok[first], val[first], key[first]


def _sample(atlas, col, k):
    # Pool poles, deduplicate, then select smallest keys, not equal counts per bin.
    # Finite pilots can omit a tiny central gap for continuous shapes; tolerances
    # below are deliberately much wider than that effect. SAE's zeros are covered.
    tok, val, key = _stored(atlas, col)
    assert len(tok) >= k
    take = key.argsort(stable=True)[:k]
    return tok[take], val[take]


@pytest.mark.parametrize("shape,rate", [("gelu", None), ("gelu_off", None),
                                       ("sae", 0.01), ("sae", 0.1)])
def test_transforms_matching_tokens_and_positive_topk(generate, shape, rate):
    gauss, columns = generate("gauss", flags=("--noise-columns", 32))
    flags = ["--shape", shape, "--noise-columns", 32]
    if rate is not None:
        flags += ["--sae-rate", rate]
    changed, other = generate(f"{shape}-{rate}", flags=flags)
    assert columns == other
    assert changed["shape"] == shape
    assert {"sae_rate", "sae_threshold", "randdir_mu", "randdir_s"} <= changed.keys()
    assert changed["randdir_mu"] is None
    threshold = float(torch.special.ndtri(torch.tensor(1 - rate))) if rate is not None else 0
    if rate is not None:
        assert changed["sae_rate"] == rate
        assert float(changed["sae_threshold"]) == pytest.approx(threshold, abs=2e-6)
        # Use the reported threshold for fp32 arithmetic; ndtri's input dtype
        # is unspecified, so its final few bits need not match this oracle.
        threshold = float(changed["sae_threshold"])

    def transform(z):
        if shape == "sae":
            return F.relu(z - threshold)
        return F.gelu(z - (2 if shape == "gelu_off" else 0), approximate="none")

    for entry in columns:
        col = entry["col"]
        before = gauss["top_val"][0, 0, col]
        # Only this pole and domain preserve rank. SAE must have no zero ties.
        assert bool((before > (threshold if shape == "sae" else 2)).all())
        assert torch.equal(gauss["top_tok"][0, 0, col], changed["top_tok"][0, 0, col])
        torch.testing.assert_close(changed["top_val"][0, 0, col], transform(before),
                                   rtol=1e-6, atol=2e-7)
        gt, gv, _ = _stored(gauss, col, top=True)
        ct, cv, _ = _stored(changed, col, top=True)
        index = torch.searchsorted(ct, gt)
        common = (index < len(ct)) & (ct[index.clamp(max=len(ct) - 1)] == gt)
        assert int(common.sum()) >= 512
        torch.testing.assert_close(cv[index[common]], transform(gv[common]), rtol=1e-6, atol=2e-7)
        if shape == "sae":
            assert bool((cv >= 0).all())
        else:
            assert bool((cv >= -0.171).all())


def _binomial_interval(n, rate):
    """Equal-tail 99.9% binomial prediction interval, without scipy."""
    k = torch.arange(n + 1, dtype=torch.float64)
    logp = (math.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
            + k * math.log(rate) + (n - k) * math.log1p(-rate))
    cdf = logp.exp().cumsum(0)
    return tuple(int(torch.searchsorted(cdf, torch.tensor(q, dtype=torch.float64)))
                 for q in (0.0005, 0.9995))


@pytest.mark.parametrize("rate", [0.01, 0.1])
def test_sae_noise_firing_rate(generate, rate):
    atlas, columns = generate(f"sae-{rate}", flags=("--shape", "sae", "--sae-rate", rate,
                                                  "--noise-columns", 32))
    # Extra columns are last and independent; omit any legacy noise column.
    noise = columns[-32:]
    assert all(c["type"] == "noise" and c["alpha"] == 0 for c in noise)
    values = torch.cat([_sample(atlas, c["col"], 2048)[1] for c in noise])
    assert values.numel() >= 50_000
    lo, hi = _binomial_interval(values.numel(), rate)
    assert lo <= int((values > 0).sum()) <= hi


def _parts(family, size):
    return [[RES_ORDER[j], 1 if family == "same" else (1, 2, 0)[j % 3]] for j in range(size)]


def _signal(data, parts, tok):
    protein, position = tok // STRIDE, tok % STRIDE
    aa = torch.tensor([AA.index(data["seq"][i][j]) for i, j in zip(protein.tolist(), position.tolist())])
    state = data["states"][protein, position]
    return sum(((aa == AA.index(residue)) & (state == code)).long() for residue, code in parts)


@pytest.fixture(scope="module")
def mixtures(generate):
    return generate("mixtures", "small", flags=("--mixtures", "--noise-columns", 3), kr=256, pilot=256)


def test_mixture_metadata_order_prefixes_and_noise_count(generate, mixtures):
    _, base = generate("default", "small", kr=256, pilot=256)
    atlas, columns = mixtures
    assert atlas["shape"] == "gauss"
    assert columns[:len(base)] == base
    ladder = sorted({(c["alpha"], c["rep"]) for c in base})
    expected = []
    for family in ("same", "diff"):
        for size in (2, 8, 20):
            expected.append((f"mix_{family}_{size}", _parts(family, size)))
    for family in ("same", "diff"):
        for j in range(20):
            expected.append((f"part_{family}_{j}", [_parts(family, 20)[j]]))
    appended = columns[len(base):-3]
    assert len(appended) == len(expected) * len(ladder)
    for entry, (kind, parts, alpha, rep) in zip(appended, [
            (kind, parts, alpha, rep) for kind, parts in expected for alpha, rep in ladder]):
        assert {"col", "type", "alpha", "rep", "parts"} <= entry.keys()
        assert (entry["type"], entry["parts"], entry["alpha"], entry["rep"]) == (kind, parts, alpha, rep)
    for i, c in enumerate(columns):
        assert c["col"] == i
    assert all(c["type"] == "noise" and c["alpha"] == 0 for c in columns[-3:])
    assert sum(c["type"] == "noise" for c in columns) == 3 + sum(c["type"] == "noise" for c in base)
    for family in ("same", "diff"):
        by_type = {c["type"]: c["parts"] for c in appended}
        assert by_type[f"mix_{family}_2"] == by_type[f"mix_{family}_8"][:2]
        assert by_type[f"mix_{family}_8"] == by_type[f"mix_{family}_20"][:8]


def test_mixture_signals_sum_and_worked_example(generate, inputs, mixtures):
    atlas, columns = mixtures
    rotated, other = generate("rotated", "rotated", flags=("--mixtures", "--noise-columns", 3),
                              kr=256, pilot=256)
    assert columns == other
    data, control = inputs["small"], inputs["rotated"]
    differences = {}
    for c in columns:
        if not c["type"].startswith(("mix_", "part_")):
            continue
        tok, values, _ = _stored(atlas, c["col"])
        rt, rv, _ = _stored(rotated, c["col"])
        # Full small pilot plus Kr >= N lets the two poles cover every token.
        assert len(tok) == data["n"] and torch.equal(tok, rt)
        signal = _signal(data, c["parts"], tok)
        null = _signal(control, c["parts"], tok)
        delta = values.double() - rv.double()
        torch.testing.assert_close(delta, c["alpha"] * (signal - null).double(),
                                   rtol=0, atol=4e-6 * max(1, c["alpha"]))
        differences[c["type"], c["alpha"], c["rep"]] = delta
        if c["type"] == "mix_same_2":
            assert signal[:4].tolist() == [1, 1, 0, 1]
        if c["type"] == "mix_diff_2":
            assert signal[:4].tolist() == [1, 0, 0, 1]
        if c["type"] in ("part_same_0", "part_same_1", "part_diff_1"):
            assert signal[:4].tolist() == {"part_same_0": [1, 0, 0, 1],
                                          "part_same_1": [0, 1, 0, 0],
                                          "part_diff_1": [0, 0, 0, 0]}[c["type"]]
    assert any(c["alpha"] > 0 for c in columns)
    for c in columns:
        if not c["type"].startswith("mix_"):
            continue
        _, family, size = c["type"].split("_")
        tok, _, _ = _stored(atlas, c["col"])
        part_signals = [_signal(data, [_parts(family, 20)[j]], tok) for j in range(int(size))]
        assert torch.equal(_signal(data, c["parts"], tok), sum(part_signals))
        part_deltas = [differences[f"part_{family}_{j}", c["alpha"], c["rep"]] for j in range(int(size))]
        torch.testing.assert_close(differences[c["type"], c["alpha"], c["rep"]], sum(part_deltas),
                                   rtol=0, atol=8e-5 * max(1, c["alpha"]))


def test_randdir_center_scale_and_signal_residuals(generate, inputs):
    atlas, columns = generate("randdir", "randdir", flags=("--shape", "randdir", "--mixtures",
                                                           "--noise-columns", 64), kr=1024)
    assert atlas["shape"] == "randdir"
    # 200-node Gauss-Hermite for standard normal via its Jacobi matrix.
    off = torch.arange(1, 200, dtype=torch.float64).sqrt()
    nodes, vectors = torch.linalg.eigh(torch.diag(off, 1) + torch.diag(off, -1))
    m1 = 1 / (2 * math.sqrt(math.pi))
    variance = float((vectors[0].square() * F.gelu(nodes, approximate="none").square()).sum()) - m1 ** 2
    scale = math.sqrt(variance)
    assert float(atlas["randdir_s"]) == pytest.approx(scale, abs=2e-6)
    mu = torch.as_tensor(atlas["randdir_mu"])
    assert mu.numel() == len(columns)
    for c in columns:
        # Spec gives a seed and unit norm, but no sampling method. Read this
        # as the usual isotropic construction: normalize 64 Gaussian draws.
        gen = torch.Generator().manual_seed(1_000_000 + c["col"])
        weights = torch.randn(64, generator=gen)
        weights /= weights.norm()
        assert float(mu[c["col"]]) == pytest.approx(m1 * float(weights.sum()), abs=2e-6)

    def normal(values):
        values = values.double()
        assert abs(float(values.mean())) < 0.05
        assert abs(float(values.std()) - 1) < 0.05

    noise = columns[-64:]
    assert all(c["type"] == "noise" and c["alpha"] == 0 for c in noise)
    values = torch.cat([_sample(atlas, c["col"], 1024)[1] for c in noise])
    assert values.numel() >= 50_000
    normal(values)
    residuals = {}
    for c in columns:
        if not c["type"].startswith(("mix_", "part_")):
            continue
        tok, value = _sample(atlas, c["col"], 1024)
        signal = _signal(inputs["randdir"], c["parts"], tok)
        residuals.setdefault(c["alpha"], []).append(value.double() - c["alpha"] * signal / scale)
    assert set(residuals) == {c["alpha"] for c in columns}
    for alpha, pieces in residuals.items():
        normal(torch.cat(pieces))


def main():
    return 0 if pytest.main([str(Path(__file__)), "-q", *sys.argv[1:]]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
