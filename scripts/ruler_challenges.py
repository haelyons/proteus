"""Challenges for the frozen ruler v1.2 (expectations: results/ruler_challenges_predictions.md).

One script builds a challenge set, streams it through the same Atlas code as model units, scores every column
with the frozen ruler (atlas/ruler.py at aab0c82, rule "v1.2"), applies the declared window baseline v0
(atlas/seqbaseline.py) to structural / context calls, and writes calls and error rates with Clopper-Pearson
intervals at stated mixes. Sets: A (fp32); B (z quantised to bf16); C (protein ids permuted: new token ids,
hashes, ties, reservoir samples; same z); D (5% of proteins duplicated with identical z, same cluster).

  python scripts/ruler_challenges.py --set A --data swissprot.pt --desc-dir desc_subset/ --uniref lists/..tsv.gz --out dir
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.core import Atlas, AtlasConfig, pack_tok  # noqa: E402
from atlas.reference import pilot_edges  # noqa: E402
from atlas.ruler import Tables, calls, score_column  # noqa: E402
from atlas.seqbaseline import WindowTables, baseline_test  # noqa: E402
from scripts.synthetic_columns import load_subset, signals, state5  # noqa: E402

NONE = "no association found by this procedure"
ASSOC = {"structural association", "composition association", "context association"}
ALPHAS_NEW = [0.5, 1.0, 2.0, 3.0, 4.0]
OLD = ["helix", "buried", "trp", "lys_helix", "two_job", "ordered", "narrow", "hydro_win"]


def column_spec():
    cols = []
    for t in ("motif_nxst", "motif_gxgxxg", "conj_exposed_strand", "balanced_xor"):
        cols += [(t, a, "iid", r) for a in ALPHAS_NEW for r in range(3)]
    cols += [(t, a, "iid", r) for t in ("hydro_win", "helix") for a in (1.0, 3.0) for r in range(3)]
    for nm in ("prot", "clus", "smooth"):
        cols += [("noise", 0.0, nm, r) for r in range(30)]
        cols += [(t, a, nm, r) for t in ("helix", "trp", "ordered") for a in (1.0, 3.0) for r in range(3)]
    cols += [("noise", 0.0, "iid", r) for r in range(150)]
    cols += [(t, a, "iid", 10 + r) for t in OLD for a in (1.0, 2.0, 3.0) for r in range(2)]
    return cols


def column_spec_e():
    """Fresh baseline challenges (cases_round2_plan.md section 2): new sequence-only and structural types."""
    cols = []
    for t in ("hp_i34", "kxxxe", "pxxp", "strand_buried", "helix_exposed"):
        cols += [(t, a, "iid", r) for a in (1.0, 2.0, 3.0, 4.0) for r in range(3)]
    cols += [(t, a, "iid", r) for t in ("helix", "hydro_win", "motif_gxgxxg") for a in (2.0, 3.0) for r in range(2)]
    cols += [("noise", 0.0, "iid", r) for r in range(60)]
    return cols


def clopper_pearson(k, n, a=0.05):
    if n == 0:
        return (None, None)

    def cdf(kk, p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(kk + 1))

    def solve(f, lo=0.0, hi=1.0):
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: cdf(k - 1, p) > 1 - a / 2)
    upper = 1.0 if k == n else solve(lambda p: cdf(k, p) > a / 2)
    return (round(lower, 4), round(upper, 4))


def build_proteins(a):
    data = torch.load(a.data, weights_only=False)
    prot = load_subset(a.desc_dir, data, a.uniref)
    g = torch.Generator().manual_seed(a.seed)
    prot = [prot[i] for i in torch.randperm(len(prot), generator=g).tolist()]
    for p in prot:
        p["noise_key"] = p["seq_idx"]                        # z depends on this, never on the token id
    if a.dup_frac > 0:                                        # set D: identical copies, same cluster, new ids
        nd = int(a.dup_frac * len(prot)); top = max(p["seq_idx"] for p in prot)
        pick = torch.randperm(len(prot), generator=g)[:nd].tolist()
        dups = [dict(prot[i], seq_idx=top + 1 + k) for k, i in enumerate(pick)]
        prot = prot + dups
        prot = [prot[i] for i in torch.randperm(len(prot), generator=g).tolist()]
    if a.relabel_seed:                                        # set C: permute protein ids (token ids) only
        ids = [p["seq_idx"] for p in prot]
        perm = torch.randperm(len(ids), generator=torch.Generator().manual_seed(a.relabel_seed)).tolist()
        for p, j in zip(prot, perm):
            p["seq_idx"] = ids[j]
    return prot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--desc-dir", required=True)
    ap.add_argument("--uniref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--noise-seed", type=int, default=5)
    ap.add_argument("--tokens-per-batch", type=int, default=32768)
    ap.add_argument("--pilot-tokens", type=int, default=1_000_000)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--max-proteins", type=int, default=0, help="smoke only")
    ap.add_argument("--Kr", type=int, default=64)
    ap.add_argument("--spec", default="A", choices=["A", "E"], help="A: the challenge set; E: fresh baseline challenges")
    ap.add_argument("--baseline", default="v0", choices=["v0", "v1", "none"])
    a = ap.parse_args()
    a.bf16 = a.set == "B"; a.relabel_seed = 77 if a.set == "C" else 0; a.dup_frac = 0.05 if a.set == "D" else 0.0
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    prot = build_proteins(a)
    if a.max_proteins:
        prot = prot[: a.max_proteins]
    cols = column_spec() if a.spec == "A" else column_spec_e(); C = len(cols)
    (out / "columns.json").write_text(json.dumps([{"col": i, "type": t, "alpha": al, "noise": nm, "rep": r} for i, (t, al, nm, r) in enumerate(cols)]))
    allb = torch.cat([p["burial"][p["plddt"] >= 70] for p in prot])
    bmu, bsd = float(allb.mean()), float(allb.std())
    cut = torch.quantile(allb, torch.tensor([1 / 3, 2 / 3]))   # exactly as atlas.ruler.Tables computes its terciles

    def burial3(p):
        b = torch.full((len(p["seq"]),), -1, dtype=torch.long)
        ok = p["plddt"] >= 70
        b[ok] = torch.bucketize(p["burial"][ok], cut, right=True)
        return b

    # balanced interaction: target joint Q = P + delta * M on state5 {loop 0, helix 1} x burial3 {exposed 0, buried 2}
    joint = torch.zeros(5, 4)                                  # burial index 3 = undefined
    for p in prot:
        s = state5(p); b = burial3(p); b[b < 0] = 3
        joint.index_put_((s, b), torch.ones(len(s)), accumulate=True)
    P = joint / joint.sum()
    M = torch.zeros(5, 4); M[1, 2] = M[0, 0] = 1.0; M[1, 0] = M[0, 2] = -1.0
    delta = 0.6 * float(min(P[1, 0], P[0, 2]))
    ratio = (P + delta * M) / P.clamp(min=1e-12)
    pi = ratio / ratio.max()                                   # selection probability per cell

    noise_rng = lambda key, salt: torch.Generator().manual_seed((int(key) * 1_000_003 + salt + (a.noise_seed << 40)) % (2 ** 63))  # noqa: E731
    kern = torch.exp(-0.5 * (torch.arange(-15, 16).float() / 5.0) ** 2); kern = kern / kern.norm()
    nmodel = [nm for (_, _, nm, _) in cols]
    m_iid = torch.tensor([n == "iid" for n in nmodel]); m_prot = torch.tensor([n == "prot" for n in nmodel])
    m_clus = torch.tensor([n == "clus" for n in nmodel]); m_smooth = torch.tensor([n == "smooth" for n in nmodel])
    xor_sel = {"n": 0, "state5": torch.zeros(5), "burial3": torch.zeros(4)}
    tnames = sorted({t for (t, al, _, _) in cols if t != "noise"})
    tix = torch.tensor([tnames.index(t) if t != "noise" else -1 for (t, _, _, _) in cols])
    amp = torch.tensor([al if t != "noise" else 0.0 for (t, al, _, _) in cols])
    namp = torch.tensor([0.5 * al if t == "narrow" else 0.0 for (t, al, _, _) in cols])

    def column_block(p):
        n = len(p["seq"]); key = p["noise_key"]
        sg = signals(p, bmu, bsd)
        s5 = state5(p); b3 = burial3(p)
        seq = p["seq"]
        nxst = torch.zeros(n); gx = torch.zeros(n)
        for m in re.finditer(r"(?=N[^P][ST])", seq):
            nxst[m.start()] = 1.0
        for m in re.finditer(r"(?=G.G..G)", seq):
            gx[m.start()] = 1.0
        conj = ((s5 == 2) & (b3 == 0)).float()
        hyd = torch.tensor([float(ch in "ILVFMW") for ch in seq])
        hp34 = torch.zeros(n); kxe = torch.zeros(n); pxp = torch.zeros(n)
        if n > 4:
            hp34[: n - 4] = hyd[: n - 4] * hyd[3: n - 1] * hyd[4:n]
        for m in re.finditer(r"(?=K...E)", seq):
            kxe[m.start()] = 1.0
        for m in re.finditer(r"(?=P..P)", seq):
            pxp[m.start()] = 1.0
        bb = b3.clone(); bb[bb < 0] = 3
        u = torch.rand(n, generator=noise_rng(key, 11))
        xor = (u < pi[s5, bb]).float()
        xor_sel["n"] += int(xor.sum()); xor_sel["state5"] += torch.bincount(s5[xor > 0], minlength=5).float()
        xor_sel["burial3"] += torch.bincount(bb[xor > 0], minlength=4).float()
        sg.update({"motif_nxst": nxst, "motif_gxgxxg": gx, "conj_exposed_strand": conj, "balanced_xor": xor,
                   "hp_i34": hp34, "kxxxe": kxe, "pxxp": pxp, "strand_buried": ((s5 == 2) & (b3 == 2)).float(),
                   "helix_exposed": ((s5 == 1) & (b3 == 0)).float()})
        E = torch.randn(n, C, generator=noise_rng(key, 1))
        up = torch.randn(C, generator=noise_rng(key, 2))
        uc = torch.randn(C, generator=noise_rng(zlib.crc32(p["cluster"].encode()), 3))
        Wn = torch.randn(n + 30, C, generator=noise_rng(key, 4))
        S = torch.nn.functional.conv1d(Wn.T[:, None, :], kern[None, None, :])[:, 0, :].T   # [n, C], unit variance
        r = 0.3
        Z = torch.where(m_iid, E, torch.zeros_like(E))
        Z = Z + torch.where(m_prot, math.sqrt(r) * up + math.sqrt(1 - r) * E, torch.zeros_like(E))
        Z = Z + torch.where(m_clus, math.sqrt(r) * uc + math.sqrt(1 - r) * E, torch.zeros_like(E))
        Z = Z + torch.where(m_smooth, S, torch.zeros_like(E))
        SG = torch.stack([sg[t] for t in tnames], 1)                       # [n, types]
        Z += SG[:, tix.clamp(min=0)] * amp[None, :] + sg["narrow_weak"][:, None] * namp[None, :]
        if a.bf16:
            Z = Z.to(torch.bfloat16).float()
        return Z

    pilot, n_tok = [], 0
    for p in prot:
        if n_tok >= a.pilot_tokens:
            break
        Z = column_block(p); pilot.append(Z); n_tok += Z.shape[0]
    xor_sel.update(n=0, state5=torch.zeros(5), burial3=torch.zeros(4))   # count the stream only
    Hp = torch.cat(pilot); del pilot
    E = pilot_edges([Hp], probs=(0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999), chunk=32)   # 32 columns per sort: 15 GB container
    del Hp
    at = Atlas(1, C, E, AtlasConfig(K=64, Kr=a.Kr, backend="torch", seed=a.seed, gram=False, cap_frac=0.02), n_annot=0)
    blocks, toks, cur, nb = [], [], 0, 0

    def flush():
        H = torch.cat(blocks); tk = torch.cat(toks); n = H.shape[0]
        ctx = at.make_ctx(n, torch.arange(n), tk, None)
        at.begin_batch(ctx); at.observe(0, H)
        redo = at.end_batch()
        if redo:
            at.begin_batch(ctx, redo_layers=redo); at.observe(0, H); at.end_batch()
    for p in prot:
        Z = column_block(p)
        blocks.append(Z); toks.append(pack_tok(torch.full((Z.shape[0],), p["seq_idx"]), torch.arange(Z.shape[0]))); cur += Z.shape[0]
        if cur >= a.tokens_per_batch:
            flush(); blocks, toks, cur = [], [], 0; nb += 1
    if blocks:
        flush(); nb += 1
    res = at.result()
    t_gen = time.time() - t0
    # the planted interaction's marginals against the background (stream)
    bgs = P.sum(1); bgb = P.sum(0)
    xs = xor_sel["state5"] / max(xor_sel["n"], 1); xb = xor_sel["burial3"] / max(xor_sel["n"], 1)
    xor_check = {"selected": xor_sel["n"], "state5_selected_minus_background": [round(float(v), 4) for v in xs - bgs],
                 "burial3_selected_minus_background": [round(float(v), 4) for v in xb - bgb], "delta": round(delta, 4)}

    T = Tables(prot)
    rt = res["res_tok"][0, 0]
    scores = [score_column(T, rt[c], B=a.B, seed=c) for c in range(C)]
    full = calls(scores, rule="v1.2")
    torch.save({"res_tok": res["res_tok"][0, 0].to(torch.int32)}, out / "res_tok.pt")      # for re-running diagnostics
    if a.baseline == "v0":
        WT = WindowTables(T, W=5, n_bg=200_000, seed=0)
    elif a.baseline == "v1":
        from atlas.seqbaseline import WindowTablesV1, baseline_v1
        WT = WindowTablesV1(T, W=5, n_bg=1_000_000, seed=0)
    for c in full:
        if c["call"] in ("structural association", "context association") and a.baseline != "none":
            if a.baseline == "v0":
                c["window_baseline_v0"] = baseline_test(WT, rt[c["col"]], c["primary"], B=a.B, seed=c["col"])
            else:
                c["window_baseline_v1"] = baseline_v1(WT, rt[c["col"]], c["primary"], B=a.B, seed=c["col"])
        s = scores[c["col"]]["peak"]
        c["peak_p"] = {k: (s[k]["p"] if s.get(k) else None) for k in ("state5", "burial3", "rama4")}
        c["n_clusters_peak"] = s.get("n_clusters", 0)
    # mixes: all null; ~80% signal (null columns subsampled); the set as built
    null_ix = [i for i, (t, al, _, _) in enumerate(cols) if t == "noise" or al == 0]
    for c in full:
        b = c.get("window_baseline_v1") or {}
        if "p" in b: b.pop("p")
    sig_ix = [i for i in range(C) if i not in set(null_ix)]
    g = torch.Generator().manual_seed(1)
    keep_null = [null_ix[j] for j in torch.randperm(len(null_ix), generator=g)[: max(1, len(sig_ix) // 4)].tolist()]
    mixes = {"as_built": list(range(C)), "all_null": null_ix, "signal_80": sorted(sig_ix + keep_null)}
    mix_res = {}
    for name, ix in mixes.items():
        cc = calls([scores[i] for i in ix], rule="v1.2")
        fc = sum(1 for k, i in enumerate(ix) if i in set(null_ix) and cc[k]["call"] in ASSOC)
        nn = sum(1 for i in ix if i in set(null_ix))
        ncall = sum(1 for c in cc if c["call"] in ASSOC)
        ninsuf = sum(1 for c in cc if c["call"] == "insufficient samples")
        mix_res[name] = {"columns": len(ix), "null_columns": nn, "insufficient_samples": ninsuf, "false_calls": fc, "false_call_rate": round(fc / nn, 4) if nn else None,
                         "false_call_rate_95ci": clopper_pearson(fc, nn), "calls": ncall,
                         "false_discovery_proportion": round(fc / ncall, 4) if ncall else None,
                         "fdp_95ci": clopper_pearson(fc, ncall) if ncall else None}
    for c in full:
        c.pop("levels", None)
    out_calls = [dict(c, **{"type": cols[c["col"]][0], "alpha": cols[c["col"]][1], "noise": cols[c["col"]][2], "rep": cols[c["col"]][3]}) for c in full]
    (out / "calls.json").write_text(json.dumps(out_calls))
    summary = {"set": a.set, "spec": a.spec, "Kr": a.Kr, "baseline": a.baseline, "proteins": len(prot), "tokens": at.n_seen, "columns": C, "batches": nb, "generate_s": round(t_gen, 1),
               "total_s": round(time.time() - t0, 1), "balanced_xor_check": xor_check, "mixes": mix_res,
               "ruler": "atlas/ruler.py @ aab0c82, rule v1.2",
               "baseline_detail": {"v0": "atlas/seqbaseline.py v0 (W 5, 200k background, lam 20)",
                                   "v1": "atlas/seqbaseline.py v1 (same level, 5 cluster folds, MLP 11x21-32-1, 1M background)", "none": "-"}[a.baseline]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "mixes"}))
    print(json.dumps(mix_res))


if __name__ == "__main__":
    main()
