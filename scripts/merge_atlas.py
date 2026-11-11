"""Merge shard.pt files in shard-index order, keeping only one incoming state alive."""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.core import Atlas, result_digests  # noqa: E402


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(obj, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def merge(paths, out: Path, device, allow_partial):
    start = time.perf_counter()
    shards = []
    reference = None
    # mmap avoids reading tensor storage in this metadata-only pass. Never retain
    # a tensor or the loaded dict across iterations (650M states are ~18 GB).
    for path in paths:
        loaded = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        try:
            i, n = loaded["shard"]
            if not isinstance(i, int) or not isinstance(n, int) or not 0 <= i < n:
                raise ValueError(f"{path}: invalid shard index {i}/{n}")
            metadata = {k: loaded[k] for k in
                        ("plan_sha256", "n_batches_plan", "n_total_plan")}
            metadata["source"] = loaded.get("source", "neurons_pre")
            metadata["args_key"] = {k: v for k, v in loaded["args_key"].items() if k != "shard"}
            if reference is None:
                reference = metadata
                n_shards = n
            else:
                if n != n_shards:
                    raise ValueError(f"{path}: N differs ({n} != {n_shards})")
                for k in reference:
                    if metadata[k] != reference[k]:
                        raise ValueError(f"{path}: {k} differs")
            shards.append({"path": str(path), "shard": [i, n],
                           "batches_done": loaded["batches_done"],
                           "n_seen": loaded["atlas"]["n_seen"],
                           "stats": dict(loaded["atlas"]["stats"]),
                           "bytes": path.stat().st_size})
        finally:
            del loaded
            gc.collect()
    shards.sort(key=lambda s: s["shard"][0])
    indices = [s["shard"][0] for s in shards]
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate shard index")
    if not allow_partial and (len(indices) != n_shards or indices != list(range(n_shards))):
        raise ValueError(f"shard indices do not cover 0..{n_shards - 1}")
    if not allow_partial and sum(s["batches_done"] for s in shards) != reference["n_batches_plan"]:
        raise ValueError("sum of batches_done differs from n_batches_plan")
    validate_s = time.perf_counter() - start

    steps = []
    for pos, shard in enumerate(shards):
        t = time.perf_counter()
        loaded = torch.load(shard["path"], map_location="cpu", weights_only=False, mmap=True)
        try:
            # from_state retains E on CPU. Detach it so deleting loaded also
            # unmaps the file instead of keeping all its faulted-in pages alive.
            loaded["atlas"]["E"] = loaded["atlas"]["E"].clone()
            incoming = Atlas.from_state(loaded["atlas"], device=device)
            if pos == 0:
                provenance = {k: loaded[k] for k in ("model", "coverage", "vocab", "order_seed")}
        finally:
            del loaded
            gc.collect()
        load_s = time.perf_counter() - t
        t = time.perf_counter()
        if pos == 0:
            acc = incoming
        else:
            # Keep merge_from's edge, config and seed assertions intact.
            acc.merge_from(incoming)
        del incoming
        gc.collect()
        steps.append({"load_s": load_s, "merge_s": time.perf_counter() - t,
                      "rss_after_bytes": peak_rss_bytes()})

    t = time.perf_counter()
    res = acc.result()
    res.update(provenance)
    res.update(n_total_plan=reference["n_total_plan"], n_batches=reference["n_batches_plan"])
    result_s = time.perf_counter() - t
    digests = result_digests(res)
    out.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    atomic_save(res, out / "atlas.pt")
    save_s = time.perf_counter() - t
    report = {"shards": shards, "steps": steps, "validate_s": validate_s,
              "validation": "metadata-only mmap pass; each input loaded twice",
              "rss_measurement": "process high-water mark after each fold",
              "result_s": result_s, "save_s": save_s,
              "total_s": time.perf_counter() - start, "peak_rss_bytes": peak_rss_bytes(),
              "digests": digests, "n_seen": res["n_seen"], "n_sub": res["n_sub"],
              "gram_rows": res.get("gram_rows"), "stats": res["stats"],
              "atlas_pt_bytes": (out / "atlas.pt").stat().st_size}
    for k in ("res_seeded_bins", "res_seed_limited_bins", "top_seed_short_groups"):
        report[k] = res[k]
    (out / "merge.json").write_text(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("shards", nargs="+", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    try:
        merge(a.shards, a.out, a.device, a.allow_partial)
    except (AssertionError, KeyError, ValueError, RuntimeError, OSError) as e:
        message = str(e) or "merge_from assertion failed (edges or config differ)"
        ap.exit(1, f"merge_atlas: {message}\n")


if __name__ == "__main__":
    main()
