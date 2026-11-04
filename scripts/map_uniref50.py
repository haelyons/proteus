"""Map every accession of our Swiss-Prot build to its UniRef50 cluster (UniProt ID mapping service,
100k accessions per job), for redundancy control: samples from one cluster are not independent.

  python scripts/map_uniref50.py --data data/swissprot_2026_03.pt --out lists/swissprot_2026_03_uniref50.tsv.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.parse
import urllib.request

import torch

UA = {"User-Agent": "esm-atlas/0.1"}
API = "https://rest.uniprot.org/idmapping"


def call(url, data=None, timeout=120):
    for attempt in range(10):
        try:
            req = urllib.request.Request(url, data=data, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode()
        except Exception:  # noqa: BLE001
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"failed: {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--parts", default=None, help="directory caching finished chunks (resume after a failure)")
    a = ap.parse_args()
    accs = list(torch.load(a.data, weights_only=False)["acc"])
    got = {}
    from pathlib import Path
    parts = Path(a.parts) if a.parts else None
    if parts:
        parts.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(accs), a.chunk):
        ids = accs[i:i + a.chunk]
        part = parts / f"chunk_{i}.tsv" if parts else None
        if part is not None and part.exists():
            for ln in part.read_text().splitlines():
                f, t = ln.split("\t")
                got[f] = t
            print(f"{min(i + a.chunk, len(accs))}/{len(accs)} (cached)", flush=True)
            continue
        body = urllib.parse.urlencode({"from": "UniProtKB_AC-ID", "to": "UniRef50", "ids": ",".join(ids)}).encode()
        job = json.loads(call(f"{API}/run", body))["jobId"]
        while json.loads(call(f"{API}/status/{job}")).get("jobStatus", "FINISHED") not in ("FINISHED",):
            time.sleep(5)
        rows = [ln.split("\t")[:2] for ln in call(f"{API}/stream/{job}?format=tsv", timeout=600).splitlines()[1:]]
        for f, t in rows:
            got[f] = t
        if part is not None:
            part.write_text("\n".join(f"{f}\t{t}" for f, t in rows))
        print(f"{min(i + a.chunk, len(accs))}/{len(accs)} mapped so far {len(got)}", flush=True)
    with gzip.open(a.out, "wt") as fo:
        fo.write("accession\tuniref50\n")
        for acc in accs:
            fo.write(f"{acc}\t{got.get(acc, '')}\n")
    n_cl = len(set(v for v in got.values()))
    print(json.dumps({"accessions": len(accs), "mapped": len(got), "unmapped": len(accs) - len(got), "clusters": n_cl}))


if __name__ == "__main__":
    main()
