"""Per-residue structural descriptors from AlphaFold DB models, for the label-free ruler.

For each UniProt accession: fetch the AFDB model (PDB format, latest version via the API), check
its sequence against our Swiss-Prot sequence, and compute per residue:
  ss3     - 3-state secondary structure (0 loop, 1 helix, 2 strand; pydssp on N, CA, C, O)
  contact - number of C-beta (C-alpha for Gly) within 8 A, |i - j| >= 3
  burial  - number of C-beta within 10 A (any separation): a buried-vs-exposed proxy
  plddt   - AlphaFold per-residue confidence (B-factor column)
  phi/psi - backbone dihedrals (degrees; NaN at chain ends); with a +-5 window they are the local
            backbone fragment descriptor
  cb      - C-beta coordinates (C-alpha for Gly), float16, for spatial relations between positions
Output: one .pt with concatenated per-residue arrays, per-protein offsets, and a status per
accession (ok / missing / fragmented / sequence-mismatch).

  python scripts/afdb_descriptors.py --data data/swissprot_2026_03.pt --n 300 --out descriptors_300.pt
  python scripts/afdb_descriptors.py --data ... --acc-file accs.txt --pdb-dir afdb_pdb/ --out ...   (local files)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import json
import os
import re
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
         "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
         "TYR": "Y", "VAL": "V", "SEC": "U", "PYL": "O"}
UA = {"User-Agent": "esm-atlas/0.1"}


def index_dir(pdb_dir: Path) -> dict:
    """acc -> path of its F1 model (latest version) in a directory of AF-<acc>-F1-model_v<N>.pdb[.gz]."""
    pat = re.compile(r"AF-(.+)-F1-model_v(\d+)\.pdb(\.gz)?$")
    best = {}
    for root, _, files in os.walk(pdb_dir):  # archives may or may not nest the models in a directory
        for name in files:
            m = pat.match(name)
            if m and (m.group(1) not in best or int(m.group(2)) > best[m.group(1)][0]):
                best[m.group(1)] = (int(m.group(2)), os.path.join(root, name))
    return {k: v[1] for k, v in best.items()}


def fetch(acc: str, pdb_dir: Path | None, cache: Path | None, path: str | None = None) -> str | None:
    if pdb_dir is not None:
        if path is None:
            return None
        return gzip.open(path, "rt").read() if path.endswith(".gz") else open(path).read()
    if cache is not None and (cache / f"{acc}.pdb").exists():
        return (cache / f"{acc}.pdb").read_text()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}", headers=UA), timeout=30) as r:
                meta = json.loads(r.read())
            if not meta:
                return None
            url = meta[0]["pdbUrl"]
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                text = r.read().decode()
            if cache is not None:
                (cache / f"{acc}.pdb").write_text(text)
            return text
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** attempt)
        except Exception:  # noqa: BLE001
            time.sleep(2 ** attempt)
    return None


def parse(text: str):
    """Backbone (N, CA, C, O), C-beta (CA for Gly) coordinates, pLDDT and sequence of chain A."""
    res = {}
    order = []
    for ln in text.splitlines():
        if not ln.startswith("ATOM"):
            continue
        name, resn, rid = ln[12:16].strip(), ln[17:20], int(ln[22:26])
        if rid not in res:
            res[rid] = {"resn": resn, "b": float(ln[60:66])}
            order.append(rid)
        res[rid][name] = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
    seq = "".join(THREE.get(res[i]["resn"], "X") for i in order)
    bb = np.full((len(order), 4, 3), np.nan, dtype=np.float32)
    cb = np.full((len(order), 3), np.nan, dtype=np.float32)
    for k, i in enumerate(order):
        r = res[i]
        for j, a in enumerate(("N", "CA", "C", "O")):
            if a in r:
                bb[k, j] = r[a]
        cb[k] = r.get("CB", r.get("CA", (np.nan,) * 3))
    plddt = np.array([res[i]["b"] for i in order], dtype=np.float32)
    return seq, bb, cb, plddt


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    x = (v * w).sum(-1)
    y = (np.cross(b1n, v) * w).sum(-1)
    return np.degrees(np.arctan2(y, x))


def descriptors(bb, cb):
    import pydssp
    n = bb.shape[0]
    ss = pydssp.assign(bb, out_type="index")                    # 0 loop, 1 helix, 2 strand
    d = np.linalg.norm(cb[:, None, :] - cb[None, :, :], axis=-1)
    sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    contact = ((d < 8.0) & (sep >= 3)).sum(1)
    burial = ((d < 10.0) & (sep >= 1)).sum(1)
    phi = np.full(n, np.nan, dtype=np.float32)
    psi = np.full(n, np.nan, dtype=np.float32)
    if n > 2:
        N, CA, C = bb[:, 0], bb[:, 1], bb[:, 2]
        phi[1:] = dihedral(C[:-1], N[1:], CA[1:], C[1:])
        psi[:-1] = dihedral(N[:-1], CA[:-1], C[:-1], N[1:])
    return np.asarray(ss, dtype=np.int8), contact.astype(np.int16), burial.astype(np.int16), phi, psi


def one(acc, seq_ref, pdb_dir, cache, path=None):
    text = fetch(acc, pdb_dir, cache, path)
    if text is None:
        return acc, "missing", None
    seq, bb, cb, plddt = parse(text)
    if seq != seq_ref:
        return acc, "sequence-mismatch", None
    ss, contact, burial, phi, psi = descriptors(bb, cb)
    return acc, "ok", {"ss3": ss, "contact": contact, "burial": burial, "plddt": plddt, "phi": phi, "psi": psi,
                       "cb": cb.astype(np.float16)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=0, help="random sample of this many accessions (0: all / --acc-file)")
    ap.add_argument("--acc-file", default=None)
    ap.add_argument("--pdb-dir", default=None, help="read AF-<acc>-F1-model_v*.pdb from here instead of downloading")
    ap.add_argument("--cache", default=None, help="keep downloaded models here")
    ap.add_argument("--threads", type=int, default=8, help="download threads (API mode)")
    ap.add_argument("--procs", type=int, default=0, help="worker processes (local files: parsing and DSSP are CPU-bound)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = torch.load(a.data, weights_only=False)
    idx = {acc: i for i, acc in enumerate(d["acc"])}
    if a.acc_file:
        accs = [ln.split()[0] for ln in open(a.acc_file) if ln.strip() and not ln.startswith("Entry")]
    else:
        accs = list(d["acc"])
    if a.n:
        g = torch.Generator().manual_seed(a.seed)
        accs = [accs[i] for i in torch.randperm(len(accs), generator=g)[: a.n].tolist()]
    pdb_dir = Path(a.pdb_dir) if a.pdb_dir else None
    cache = Path(a.cache) if a.cache else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    status, recs = {}, {}
    paths = index_dir(pdb_dir) if pdb_dir is not None else {}
    if pdb_dir is not None:
        print(f"indexed {len(paths)} F1 models in {pdb_dir}", flush=True)
    pool = cf.ProcessPoolExecutor(a.procs) if a.procs else cf.ThreadPoolExecutor(a.threads)
    with pool as ex:
        futs = [ex.submit(one, acc, d["seq"][idx[acc]], pdb_dir, cache, paths.get(acc)) for acc in accs]
        for k, f in enumerate(cf.as_completed(futs)):
            acc, st, rec = f.result()
            status[acc] = st
            if rec is not None:
                recs[acc] = rec
            if (k + 1) % (100 if len(accs) < 20000 else 20000) == 0:
                print(f"{k + 1}/{len(accs)} done, {sum(v == 'ok' for v in status.values())} ok, {time.time() - t0:.0f}s", flush=True)
    keep = [acc for acc in accs if acc in recs]
    lens = [len(recs[acc]["ss3"]) for acc in keep]
    out = {"acc": keep, "seq_idx": torch.tensor([idx[acc] for acc in keep]), "offset": torch.tensor([0] + list(np.cumsum(lens))),
           "status": status, "source": "AlphaFold DB (latest model version via the API)", "fields": ["ss3", "contact", "burial", "plddt", "phi", "psi", "cb"]}
    for k in out["fields"]:
        out[k] = torch.from_numpy(np.concatenate([recs[acc][k] for acc in keep])) if keep else torch.empty(0)
    torch.save(out, a.out)
    counts = {s: sum(v == s for v in status.values()) for s in set(status.values())}
    print(json.dumps({"n": len(accs), "status": counts, "residues": int(sum(lens)), "seconds": round(time.time() - t0, 1),
                      "ss3_frac": [round(float((out["ss3"] == c).float().mean()), 3) for c in range(3)] if keep else None,
                      "plddt_mean": round(float(out["plddt"].mean()), 1) if keep else None}))


if __name__ == "__main__":
    main()
