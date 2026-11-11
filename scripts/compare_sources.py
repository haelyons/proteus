"""Sample, score and compare atlas sources with the frozen ruler v1.2."""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.compare import column_stats, summarise  # noqa: E402
from atlas.ruler import Tables, calls, score_column  # noqa: E402
from scripts.score_real_units import load_proteins  # noqa: E402

G = {}


def sha256(path):
    path = Path(path)
    h = hashlib.sha256()
    files = [path] if path.is_file() else sorted(path.glob("part_*.pt"))
    if not files:
        raise ValueError(f"no descriptor parts: {path}")
    for fp in files:
        if path.is_dir():
            h.update(fp.name.encode())
        with fp.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def sample(a):
    if a.per_layer < 0:
        raise ValueError("per-layer must be nonnegative")
    sources = {}
    for entry in a.atlas:
        name, path = entry.split("=", 1)
        if not name or Path(name).name != name or name in sources:
            raise ValueError(f"invalid or duplicate source: {name}")
        at = torch.load(path, map_location="cpu", weights_only=False)
        L, _, D, _, _ = at["res_tok"].shape
        if a.per_layer > D:
            raise ValueError(f"per-layer exceeds {name}'s {D} columns")
        columns = []
        for layer in range(L):
            gen = torch.Generator().manual_seed(a.seed * 7919 + layer)
            columns.extend([layer, unit] for unit in torch.randperm(D, generator=gen)[:a.per_layer].tolist())
        sources[name] = {"atlas": path, "atlas_sha256": sha256(path), "L": L, "D": D, "columns": columns}
    Path(a.out).write_text(json.dumps({"seed": a.seed, "per_layer": a.per_layer, "sources": sources}, indent=2))


def work(c):
    torch.set_num_threads(1)
    return c, score_column(G["T"], G["rt"][c], B=G["B"], seed=c)


def score(a):
    if a.B < 1 or a.workers < 1:
        raise ValueError("B and workers must be positive")
    t0 = time.time()
    src = json.loads(Path(a.sample).read_text())["sources"][a.source]
    if sha256(a.atlas) != src["atlas_sha256"]:
        raise ValueError("atlas sha256 differs from sample")
    at = torch.load(a.atlas, map_location="cpu", weights_only=False)
    rt = at["res_tok"][:, 0].long()
    L, D, nb, kr = rt.shape
    if (L, D) != (src["L"], src["D"]):
        raise ValueError("atlas dimensions differ from sample")
    cols = [layer * D + unit for layer, unit in src["columns"]]
    if any(not (0 <= layer < L and 0 <= unit < D) for layer, unit in src["columns"]) or len(set(cols)) != len(cols):
        raise ValueError("sample contains invalid or duplicate columns")
    tables_hash = sha256(a.desc)
    data = torch.load(a.data, map_location="cpu", weights_only=False)
    prot = load_proteins(a.desc, data, a.uniref)
    del data
    torch.set_num_threads(1)
    G["T"] = Tables(prot); G["B"] = a.B
    G["rt"] = rt.reshape(L * D, nb, kr)
    del prot
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = dict(pool.imap_unordered(work, cols, chunksize=16))
    scores = [res[c] for c in cols]
    records = []
    for (layer, unit), c, s in zip(src["columns"], cols, scores):
        records.append({"source": a.source, "layer": layer, "unit": unit, "col": c, "seed": c, "pole": 0,
                        **column_stats(s), "score_sha256": hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.source}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    torch.save(scores, out / f"{a.source}.scores.pt")
    meta = {"B": a.B, "tables_sha256": tables_hash, "ruler_sha256": sha256(Path(__file__).resolve().parents[1] / "atlas/ruler.py"),
            "seconds": time.time() - t0, "workers": a.workers}
    (out / f"{a.source}.meta.json").write_text(json.dumps(meta, indent=2))


def summary(a):
    root = Path(a.scores)
    sources, metadata, records = {}, {}, {}
    for fp in sorted(root.glob("*.jsonl")):
        name = fp.stem
        records[name] = [json.loads(line) for line in fp.read_text().splitlines() if line.strip()]
        sources[name] = summarise(records[name])
        metadata[name] = json.loads((root / f"{name}.meta.json").read_text())
    if not sources:
        raise ValueError("no source records")
    common = {k: next(iter(metadata.values()))[k] for k in ("tables_sha256", "ruler_sha256", "B")}
    if any(any(meta[k] != v for k, v in common.items()) for meta in metadata.values()):
        raise ValueError("sources must share tables_sha256, ruler_sha256 and B")
    pool = a.pool.split(",")
    if len(set(pool)) != len(pool) or any(name not in sources for name in pool):
        raise ValueError("pool must contain distinct known sources")
    pooled, owners = [], []
    for name in pool:
        scores = torch.load(root / f"{name}.scores.pt", map_location="cpu", weights_only=False)
        if len(scores) != len(records[name]):
            raise ValueError(f"score and record counts differ: {name}")
        pooled.extend(scores)
        owners.extend([name] * len(scores))
    counts = {name: Counter() for name in pool}
    for name, c in zip(owners, calls(pooled, rule="v1.2")):
        counts[name][c["call"]] += 1
    result = {"sources": sources, "pooled_v12": {"pool": pool, "n_columns": len(pooled),
                                               "calls": {name: dict(cnt) for name, cnt in counts.items()}}, **common}
    Path(a.out).write_text(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("sample")
    sp.add_argument("--atlas", nargs="+", action="extend", required=True)
    sp.add_argument("--per-layer", type=int, required=True)
    sp.add_argument("--seed", type=int, required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(run=sample)
    sp = sub.add_parser("score")
    for arg in ("sample", "source", "atlas", "desc", "data", "uniref", "out"):
        sp.add_argument(f"--{arg}", required=True)
    sp.add_argument("--B", type=int, default=2000)
    sp.add_argument("--workers", type=int, default=4)
    sp.set_defaults(run=score)
    sp = sub.add_parser("summarise")
    for arg in ("scores", "pool", "out"):
        sp.add_argument(f"--{arg}", required=True)
    sp.set_defaults(run=summary)
    a = ap.parse_args()
    a.run(a)


if __name__ == "__main__":
    main()
