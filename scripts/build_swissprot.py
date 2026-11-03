"""Build the Swiss-Prot dataset: sequences + coarse per-residue annotations as intervals.

Output (torch.save dict) at --out:
  acc        list[str]            accession per sequence (index = seq_idx)
  seq        list[str]            full sequence (untruncated)
  len        int32 [N]            full length
  iv         int32 [M, 4]         (seq_idx, start0, end0_exclusive, ann_id) on full-length coords
  vocab      list[str]            annotation names; ids < 20 + 1 are amino-acid identity
  release    str
Truncation (to max_res residues) is applied at batch time; coverage is computed here too.

Coarse vocabulary (~150 columns):
  aa:<A..Y> (20) + aa:other; ft:<TYPE> for each UniProt feature type requested;
  ft:BINDING:<ligand> for the 40 most frequent ligands; ft:DOMAIN:<name> for the 40 most frequent
  domain names. DISULFID / CROSSLNK annotate their two end residues only.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import io
import json
import re
import sys
import time
import urllib.request

FT_FIELDS = {
    "ft_act_site": "ACT_SITE", "ft_binding": "BINDING", "ft_site": "SITE", "ft_disulfid": "DISULFID",
    "ft_mod_res": "MOD_RES", "ft_carbohyd": "CARBOHYD", "ft_lipid": "LIPID", "ft_crosslnk": "CROSSLNK",
    "ft_signal": "SIGNAL", "ft_transit": "TRANSIT", "ft_transmem": "TRANSMEM", "ft_intramem": "INTRAMEM",
    "ft_helix": "HELIX", "ft_strand": "STRAND", "ft_turn": "TURN", "ft_coiled": "COILED",
    "ft_compbias": "COMPBIAS", "ft_motif": "MOTIF", "ft_zn_fing": "ZN_FING", "ft_dna_bind": "DNA_BIND",
    "ft_domain": "DOMAIN", "ft_repeat": "REPEAT", "ft_region": "REGION", "ft_topo_dom": "TOPO_DOM",
    "ft_propep": "PROPEP", "ft_peptide": "PEPTIDE",
}
ENDPOINT_ONLY = {"DISULFID", "CROSSLNK"}
AA = "ACDEFGHIKLMNPQRSTVWY"
FEAT_RE = re.compile(r'(?:^|; )([A-Z_]+) ([<>?]?\d+|\?)(?:\.\.([<>?]?\d+|\?))?((?:; /[a-z_]+="[^"]*")*)')
QUAL_RE = re.compile(r'/([a-z_]+)="([^"]*)"')


def parse_pos(x):
    if x is None or "?" in x:
        return None
    return int(x.lstrip("<>"))


def stream(url):
    req = urllib.request.Request(url, headers={"User-Agent": "esm-atlas/0.1"})
    raw = urllib.request.urlopen(req, timeout=600)
    return io.TextIOWrapper(gzip.GzipFile(fileobj=raw), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N entries only (testing)")
    ap.add_argument("--max-res", type=int, default=1022)
    ap.add_argument("--top-ligands", type=int, default=40)
    ap.add_argument("--top-domains", type=int, default=40)
    a = ap.parse_args()
    import torch

    fields = ["accession", "length", "sequence"] + list(FT_FIELDS)
    q = "reviewed:true"
    url = (f"https://rest.uniprot.org/uniprotkb/stream?query={q}&format=tsv&compressed=true"
           f"&fields={','.join(fields)}")
    if a.limit:
        url = url.replace("/stream?", f"/search?size={a.limit}&")
    t0 = time.time()
    f = stream(url)
    header = f.readline().rstrip("\n").split("\t")
    col = {h: i for i, h in enumerate(header)}
    acc, seqs, raw_feats = [], [], []
    for line in f:
        r = line.rstrip("\n").split("\t")
        acc.append(r[0])
        seqs.append(r[2])
        feats = []
        for fname, ftype in FT_FIELDS.items():
            k = fields.index(fname)  # TSV columns follow the requested field order
            txt = r[k] if len(r) > k else ""
            if not txt:
                continue
            for m in FEAT_RE.finditer(txt):
                if m.group(1) != ftype:
                    continue
                s, e = parse_pos(m.group(2)), parse_pos(m.group(3)) if m.group(3) else parse_pos(m.group(2))
                if s is None or e is None:
                    continue
                quals = dict(QUAL_RE.findall(m.group(4) or ""))
                feats.append((ftype, s, e, quals.get("ligand") if ftype == "BINDING" else quals.get("note") if ftype == "DOMAIN" else None))
        raw_feats.append(feats)
        if a.limit and len(acc) >= a.limit:
            break
    print(f"downloaded {len(acc)} entries in {time.time()-t0:.0f}s", file=sys.stderr)

    lig = collections.Counter(x[3] for fs in raw_feats for x in fs if x[0] == "BINDING" and x[3])
    dom = collections.Counter(x[3] for fs in raw_feats for x in fs if x[0] == "DOMAIN" and x[3])
    vocab = [f"aa:{c}" for c in AA] + ["aa:other"] + [f"ft:{t}" for t in FT_FIELDS.values()]
    vocab += [f"ft:BINDING:{n}" for n, _ in lig.most_common(a.top_ligands)]
    vocab += [f"ft:DOMAIN:{n}" for n, _ in dom.most_common(a.top_domains)]
    vid = {v: i for i, v in enumerate(vocab)}

    iv = []
    for i, fs in enumerate(raw_feats):
        n = len(seqs[i])
        for ftype, s, e, name in fs:
            s0, e1 = max(0, s - 1), min(n, e)
            if e1 <= s0:
                continue
            ids = [vid[f"ft:{ftype}"]]
            if name and f"ft:{ftype}:{name}" in vid:
                ids.append(vid[f"ft:{ftype}:{name}"])
            spans = [(s0, s0 + 1), (e1 - 1, e1)] if ftype in ENDPOINT_ONLY else [(s0, e1)]
            for a0, a1 in spans:
                for k in ids:
                    iv.append((i, a0, a1, k))
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.int32)
    trunc = (lens > a.max_res)
    cov = {
        "n_seq": len(seqs), "n_res": int(lens.sum()),
        "n_truncated": int(trunc.sum()), "frac_seq_truncated": float(trunc.float().mean()),
        "res_dropped": int((lens - a.max_res).clamp(min=0).sum()),
        "frac_res_dropped": float((lens - a.max_res).clamp(min=0).sum() / lens.sum()),
        "max_len": int(lens.max()), "max_res": a.max_res,
    }
    out = {"acc": acc, "seq": seqs, "len": lens, "iv": torch.tensor(iv, dtype=torch.int32),
           "vocab": vocab, "coverage": cov, "url": url}
    torch.save(out, a.out)
    print(json.dumps({"coverage": cov, "n_intervals": len(iv), "vocab_size": len(vocab),
                      "top_ligands": lig.most_common(8), "top_domains": dom.most_common(8)}, indent=1))


if __name__ == "__main__":
    main()
