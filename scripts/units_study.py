"""Frozen unit study: sample, heldout_values, columns, report, and unit_writes reproduction.

All interventions use atlas.intervene; discovery material never selects held-out proteins.
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import bisect
import hashlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.columns import parse_sources  # noqa: E402
from atlas.compare import share_ci  # noqa: E402
from atlas.core import SEQ_STRIDE  # noqa: E402
from atlas.esm import MODELS  # noqa: E402
from scripts.pair_balance import proteins  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
SEED = 20261112
POPS = ("neurons", "directions")
G = {}


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def generator(seed):
    return torch.Generator().manual_seed(seed)


def key(c):
    return f"{c[0]}_{c[1]}_{c[2]}"


def others(c, count, offset):
    pop, l, k = c
    width = 1280 if pop == "neurons" else 512
    order = torch.randperm(width - 1, generator=generator(SEED + offset + 1000 * l + k))[:count]
    return [(pop, l, int(i) + (int(i) >= k)) for i in order]


def selected(sample):
    return [(p, l, k) for p in POPS for l, k in sample[p]]


def calibration_run(calib):
    for path in (Path(calib) / "atlases/run.json", Path(calib) / "trained/run.json"):
        if path.exists():
            return json.loads(path.read_text())
    return None


def sample(a):
    ats = [load(Path(a.calib) / "atlases" / name) for name in
           ("atlas_neurons_post.pt", "atlas_randdir_512_0.pt")]
    L = ats[0]["edges"].shape[0]
    if L != 6 or ats[0]["edges"].shape[2] != 1280 or ats[1]["edges"].shape[2] != 512:
        raise ValueError("expected ESM-2 8M calibration atlases")
    result = {}
    for p, width, offset in (("neurons", 1280, 0), ("directions", 512, 100)):
        result[p] = [[l, k] for l in range(L) for k in torch.randperm(
            width, generator=generator(SEED * 10 + offset + l))[:2 if a.smoke else 25].tolist()]
    source = parse_sources("randdir:512:0", 1280)[0]
    result["directions_sha256"] = hashlib.sha256(b"".join(
        source.directions(l, "cpu", torch.float32).numpy().tobytes() for l in range(L))).hexdigest()
    # run_atlas records this in run.json, alongside the calibration atlas.
    run = calibration_run(a.calib)
    if run is None:
        raise FileNotFoundError("calibration run.json needed for recorded directions SHA256")
    result["calib_directions_sha256"] = run["sources"]["randdir_512_0"]["directions_sha256"]
    write_json(Path(a.out) / "sample.json", result)


def model_material():
    """Only use the frozen Intervener API; obtain tokenizer/write weights independently."""
    from transformers import EsmForMaskedLM, EsmTokenizer
    tok = EsmTokenizer.from_pretrained(MODELS["8M"], local_files_only=True)
    model = EsmForMaskedLM.from_pretrained(MODELS["8M"], local_files_only=True).float().eval()
    weights = [lay.output.dense.weight.detach().clone() for lay in model.esm.encoder.layer]
    return tok, weights


def make_intervener():
    from atlas.intervene import Intervener
    return Intervener(model="8M", device="cpu", dtype=torch.float32)


def vector(c, source):
    if c[0] == "directions":
        return source.directions(c[1], "cpu", torch.float32)[:, c[2]]
    return torch.nn.functional.one_hot(torch.tensor(c[2]), 1280).float()


def heldout_values(a):
    s = json.loads((Path(a.out) / "sample.json").read_text())
    targets = selected(s)
    needed = set(targets)
    for c in targets:
        needed.update(others(c, 20, 13))
        needed.update(others(c, 2, 7))
    needed = sorted(needed)
    prot = proteins(a, True, 40 if a.smoke else 1000, 1)
    if not prot:
        raise ValueError("no held-out proteins")
    offsets = [0]
    for p in prot:
        offsets.append(offsets[-1] + len(p["seq"]))
    root = Path(a.scratch) / "values"
    root.mkdir()
    # Disk-backed columns keep the full study's held-out activations out of RAM.
    maps = {c: torch.from_file(str(root / (key(c) + ".bin")), shared=True,
                               size=offsets[-1], dtype=torch.float32) for c in needed}
    source = parse_sources("randdir:512:0", 1280)[0]
    columns = {c: (c[1], vector(c, source)) for c in needed}
    tok, _ = model_material()
    M = make_intervener()
    for j, p in enumerate(prot):
        ids = tok(p["seq"], return_tensors="pt")["input_ids"]
        run = M.run(ids, capture=True)
        for c in needed:
            maps[c][offsets[j]:offsets[j + 1]] = M.column_values(run, columns[c])
    torch.save({"proteins": prot, "offsets": offsets, "needed": needed}, Path(a.scratch) / "heldout.pt")
    write_json(Path(a.out) / "heldout_values.json", {"n_proteins": len(prot),
               "n_residues": offsets[-1], "n_columns": len(needed), "forwards": len(prot)})


def jsd(p, q):
    p, q = torch.as_tensor(p, dtype=torch.float64), torch.as_tensor(q, dtype=torch.float64)
    if p.sum() == 0 or q.sum() == 0:
        return None
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return float(sum((x[x > 0] * torch.log2(x[x > 0] / m[x > 0])).sum() for x in (p, q)) / 2)


def profile_distance(p, q):
    distances = [jsd(p[i], q[i]) for i in range(3)]
    return None if None in distances else sum(distances) / 3


def rank_pass(own, controls, higher=False):
    return all(own > x if higher else own < x for x in controls)


def profile_check(held, own, controls):
    held, own, *controls = [[p[k] for k in ("residue", "ss3", "burial")]
                            for p in (held, own, *controls)]
    d = profile_distance(held, own)
    other = [profile_distance(held, p) for p in controls]
    valid = d is not None and all(x is not None for x in other)
    return {"distance": d, "other_distances": other,
            "p": (1 + sum(x <= d for x in other)) / (len(other) + 1) if valid else None,
            "pass": bool(valid and rank_pass(d, other))}


def average_ranks(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    order = x.argsort(stable=True)
    ranks = torch.empty_like(x)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2
        i = j
    return ranks


def spearman(x, y):
    x, y = average_ranks(x), average_ranks(y)
    x, y = x - x.mean(), y - y.mean()
    norm = x.norm() * y.norm()
    return float((x @ y) / norm) if norm > 0 else None


def write_check(b, e, controls):
    own = spearman(b, e)
    other = [spearman(v, e) for v in controls]
    valid = own is not None and all(x is not None for x in other)
    return {"correlation": own, "other_correlations": other,
            "p": (1 + sum(x >= own for x in other)) / (len(other) + 1) if valid else None,
            "pass": bool(valid and rank_pass(own, other, higher=True))}


def bootstrap_diff(target, control, paired=True, draws=2000, seed=0, level=0.95):
    t, c = torch.tensor(target, dtype=torch.float64), torch.tensor(control, dtype=torch.float64)
    if paired and len(t) != len(c):
        raise ValueError("paired bootstrap requires matching positions")
    g = generator(seed)
    idx = torch.randint(len(t), (draws, len(t)), generator=g)
    ci = idx if paired else torch.randint(len(c), (draws, len(c)), generator=g)
    means = t[idx].mean(1) - c[ci].mean(1)
    quantiles = [.025, .975] if level == .95 else [(1 - level) / 2, (1 + level) / 2]
    return torch.quantile(means, torch.tensor(quantiles, dtype=torch.float64)).tolist()


def removal_check(effects):
    controls = {}
    for name in ("other", "matched", "random"):
        target, control = effects["target"], effects[name]
        if not len(target) or not len(control):
            controls[name] = {"mean": None, "ci": None, "pass": False}
            continue
        interval = bootstrap_diff(target, control, paired=name != "random")
        t, c = torch.tensor(target, dtype=torch.float64), torch.tensor(control, dtype=torch.float64)
        controls[name] = {"mean": float(t.mean() - c.mean()), "ci": interval, "pass": interval[0] > 0}
    return {"controls": controls, "pass": all(c["pass"] for c in controls.values())}


def init_worker(args):
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    a = argparse.Namespace(**args)
    G["a"] = a
    G["held"] = load(Path(a.scratch) / "heldout.pt")
    G["ats"] = {p: load(Path(a.calib) / "atlases" / name) for p, name in zip(POPS,
                ("atlas_neurons_post.pt", "atlas_randdir_512_0.pt"))}
    G["source"] = parse_sources("randdir:512:0", 1280)[0]
    G["tok"], G["weights"] = model_material()
    G["M"] = make_intervener()
    G["values"] = {}
    run = calibration_run(a.calib)
    G["edge_q"] = run["args"]["edge_q"] if run else [.5, .9, .99, .999, .9999, .99999]
    if any(at["edges"].shape[-1] != len(G["edge_q"]) for at in G["ats"].values()):
        raise ValueError("atlas edges do not match calibration quantile labels")


def values(c):
    if c not in G["values"]:
        G["values"][c] = torch.from_file(str(Path(G["a"].scratch) / "values" / (key(c) + ".bin")),
                                        size=G["held"]["offsets"][-1], dtype=torch.float32)
    return G["values"][c]


def threshold(c, q):
    at = G["ats"][c[0]]
    qs = G["edge_q"]
    return float(at["edges"][c[1], 0, c[2], list(qs).index(q)])


def positions(c):
    h, offsets = values(c), G["held"]["offsets"]
    peaks = []
    eligible_offsets = [0]
    q = threshold(c, .999)
    for j, p in enumerate(G["held"]["proteins"]):
        hits = torch.nonzero(h[offsets[j] + 3:offsets[j + 1] - 3] >= q).flatten() if len(p["seq"]) > 6 else []
        if len(hits) and len(peaks) < 60:
            peaks.append((j, int(hits[0]) + 3))
        eligible_offsets.append(eligible_offsets[-1] + max(0, len(p["seq"]) - 6))
    excluded = set(peaks)
    g = generator(SEED + 1000 * c[1] + c[2])
    n = 20 if G["a"].smoke else 60
    random = []
    for flat in torch.randperm(eligible_offsets[-1], generator=g)[:n + len(peaks)].tolist():
        j = bisect.bisect_right(eligible_offsets, flat) - 1
        pair = (j, flat - eligible_offsets[j] + 3)
        if pair not in excluded:
            random.append(pair)
        if len(random) == n:
            break
    return peaks, random


def baseline(j, i):
    tok = G["tok"]
    ids = tok(G["held"]["proteins"][j]["seq"], return_tensors="pt")["input_ids"]
    ids[0, i + 1] = tok.mask_token_id
    return ids, G["M"].run(ids, capture=True)


def direct(c):
    M = G["M"]
    r = vector(c, G["source"])
    w = G["weights"][c[1]] @ r
    out = []
    for j, i in positions(c)[1]:
        # Many columns draw the same random token. Cache only that token's
        # residual and FFN rows, never whole captured protein runs.
        path = Path(G["a"].scratch) / "direct_baselines" / f"{j}_{i}.pt"
        if path.exists():
            cached = load(path)
        else:
            _, base = baseline(j, i)
            cached = {"post": torch.stack([base.post[l][i] for l in range(6)]),
                      "x_final": base.x_final[i:i + 1].clone()}
            cached["head"] = M.head(cached["x_final"])
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            torch.save(cached, tmp)
            os.replace(tmp, path)
        av = cached["post"][c[1]] @ r
        out.append((M.head(cached["x_final"] + (threshold(c, .999) - av) * w) - cached["head"])[0])
    if not out:
        raise ValueError("no eligible random positions")
    b = torch.stack(out).mean(0).tolist()
    write_json(Path(G["a"].scratch) / "direct" / (key(c) + ".json"), b)
    return key(c)


def distributions(tokens, lookup):
    counts = {"residue": [0] * 20, "ss3": [0] * 4, "burial": [0] * 4}
    for t in tokens:
        if t < 0:
            continue
        si, i = divmod(int(t), SEQ_STRIDE)
        row = lookup.get(si)
        aa = (row[0]["seq"] if row is not None and "seq" in row[0] else G["data"]["seq"][si])[i]
        if aa in AA:
            counts["residue"][AA.index(aa)] += 1
        ss, bur = 3, None
        if row is not None:
            d, s, e = row
            if s + i < e:
                ss = int(d["ss3"][s + i])
                if ss not in (0, 1, 2):
                    ss = 3
                if float(d["plddt"][s + i]) < 70:
                    bur = 3
                elif float(d["plddt"][s + i]) >= 70 and torch.isfinite(d["burial"][s + i]):
                    bur = int(torch.bucketize(d["burial"][s + i].float(), G["cuts"], right=True))
        counts["ss3"][ss] += 1
        if bur is not None:
            counts["burial"][bur] += 1
    return {k: [v / sum(vs) if sum(vs) else 0 for v in vs] for k, vs in counts.items()}


def discovery(c):
    """Pool top tokens and the peak reservoir, retaining their multiplicities."""
    at = G["ats"][c[0]]
    top = at["top_tok"][c[1], 0, c[2]][:64].tolist()
    reservoir = at["res_tok"][c[1], 0, c[2], -1].tolist()
    return distributions(top + reservoir, G["lookup"])


def token_examples(tokens, lookup):
    examples = []
    for t in tokens:
        if t < 0:
            continue
        si, i = divmod(t, SEQ_STRIDE)
        seq = G["data"]["seq"][si]
        profile = distributions([t], lookup)
        examples.append({"token": t, "acc": G["data"]["acc"][si], "pos": i,
                         "residue": seq[i], "context": seq[max(0, i - 3):i + 4],
                         "ss3": profile["ss3"].index(1.),
                         "burial": profile["burial"].index(1.) if sum(profile["burial"]) else None})
    return examples


def prepare_profiles(a):
    # Whole-build descriptors supply atlas-token annotations; only subset discovery
    # proteins supply burial cut points.
    data = load(a.data)
    burial = []
    for path in sorted(Path(a.desc_dir).glob("part_*.pt")):
        part = load(path)
        for j, heldout in enumerate(part["heldout"]):
            if not heldout:
                s, e = int(part["offset"][j]), int(part["offset"][j + 1])
                b = part["burial"][s:e].float()
                burial.append(b[(part["plddt"][s:e] >= 70) & torch.isfinite(b)])
    burial = torch.cat(burial)
    cuts = torch.quantile(burial[torch.isfinite(burial)], torch.tensor([1 / 3, 2 / 3]))
    torch.save(cuts, Path(a.scratch) / "cuts.pt")
    d = load(a.desc)
    lookup = {int(si): (d, int(d["offset"][j]), int(d["offset"][j + 1]))
              for j, si in enumerate(d["seq_idx"])}
    G.update(data=data, lookup=lookup, cuts=cuts)
    s = json.loads((Path(a.out) / "sample.json").read_text())
    needed = set(selected(s))
    for c in selected(s):
        needed.update(others(c, 20, 13))
    profiles = {key(c): discovery(c) for c in needed}
    cards = {}
    for c in selected(s):
        tokens = G["ats"][c[0]]["top_tok"][c[1], 0, c[2]][:64].tolist()
        reservoir = G["ats"][c[0]]["res_tok"][c[1], 0, c[2], -1].tolist()
        cards[key(c)] = {"top64": token_examples(tokens, lookup),
                        "peak_reservoir": token_examples(reservoir, lookup),
                        "labels": {"residue": list(AA), "ss3": ["loop", "helix", "strand", "missing"],
                                   "burial": ["exposed", "middle", "buried", "pLDDT < 70"]},
                        "burial_cuts": cuts.tolist(), "distributions": profiles[key(c)],
                        "top64_distributions": distributions(tokens, lookup),
                        "peak_reservoir_distributions": distributions(reservoir, lookup)}
    write_json(Path(a.scratch) / "profiles.json", profiles)
    write_json(Path(a.scratch) / "discovery.json", cards)
    G.clear()


def kl20(base, edited):
    b = torch.log_softmax(base.double(), -1)
    e = torch.log_softmax(edited.double(), -1)
    return float((b.exp() * (b - e)).sum().clamp(min=0))


def column(c):
    a, M = G["a"], G["M"]
    root = Path(a.scratch)
    peers = others(c, 20, 13)
    controls = others(c, 2, 7)
    r = vector(c, G["source"])
    col = (c[1], r)
    norm = float((G["weights"][c[1]] @ r).norm())
    peaks, random = positions(c)
    effects = {kind: {k: [] for k in ("target", "other", "matched", "random")}
               for kind in ("median", "resample")}
    later = {l: torch.zeros(1280, dtype=torch.float64) for l in range(c[1] + 1, 6)}
    steering = []
    # Draw resampling values uniformly over held-out tokens, independently of
    # random-position and control selection generators.
    gen = generator(SEED + 19 + 1000 * c[1] + c[2])
    for is_peak, pairs in ((True, peaks), (False, random)):
        for j, i in pairs:
            ids, base = baseline(j, i)
            av = float(M.column_values(base, col)[i])
            if not is_peak:
                res = M.replace(ids, col, [i], [threshold(c, .999)], baseline=base)
                steering.append(res.dlp[i])
            for kind in effects:
                vals = [threshold(c, .5)] if kind == "median" else values(c)[
                    torch.randint(len(values(c)), (3,), generator=gen)].tolist()
                measured = {k: [] for k in effects[kind]}
                for v in vals:
                    if is_peak and kind == "median":
                        # Need all later units before truncation to the top ten.
                        edited = M.run(ids, edit=(c[1], r, [i], [v]), capture=True)
                        delta = edited.lp[i] - base.lp[i]
                        for l in later:
                            later[l] += (edited.post[l] - base.post[l]).double().abs().sum(0)
                    else:
                        delta = M.replace(ids, col, [i], [v], baseline=base).dlp[i]
                    measured["target" if is_peak else "random"].append(kl20(base.lp[i], base.lp[i] + delta))
                    if is_peak:
                        for name, cc in zip(("other", "matched"), controls):
                            cr = vector(cc, G["source"])
                            cv = threshold(cc, .5) if kind == "median" else float(values(cc)[
                                int(torch.randint(len(values(cc)), (1,), generator=gen))])
                            if name == "matched":
                                ca = float(M.column_values(base, (cc[1], cr))[i])
                                cn = float((G["weights"][cc[1]] @ cr).norm())
                                cv = ca + (v - av) * norm / cn
                            delta_c = M.replace(ids, (cc[1], cr), [i], [cv], baseline=base).dlp[i]
                            measured[name].append(kl20(base.lp[i], base.lp[i] + delta_c))
                for name, xs in measured.items():
                    if xs:
                        effects[kind][name].append(sum(xs) / len(xs))
    profiles = json.loads((root / "profiles.json").read_text())
    discovery_card = json.loads((root / "discovery.json").read_text())[key(c)]
    prot = G["held"]["proteins"]
    held_lookup = {p["seq_idx"]: (p, 0, len(p["seq"])) for p in prot}
    G["cuts"] = load(root / "cuts.pt")
    tokens = [prot[j]["seq_idx"] * SEQ_STRIDE + i for j, i in peaks]
    held_profile = distributions(tokens, held_lookup)
    pc = profile_check(held_profile, profiles[key(c)], [profiles[key(p)] for p in peers])
    b = json.loads((root / "direct" / (key(c) + ".json")).read_text())
    e = torch.stack(steering).mean(0).tolist()
    wc = write_check(b, e, [json.loads((root / "direct" / (key(p) + ".json")).read_text()) for p in peers])
    rc = {kind: removal_check(effects[kind]) for kind in effects}
    insufficient = len(peaks) < (4 if a.smoke else 10)
    if insufficient:
        pc["pass"] = None
        for x in rc.values():
            x["pass"] = None
    ordered = sorted(((l, u, float(v)) for l, vs in later.items() for u, v in enumerate(vs)),
                     key=lambda t: (-t[2], t[0], t[1]))
    total = sum(t[2] for t in ordered)
    card = {"population": c[0], "layer": c[1], "column": c[2], "insufficient": insufficient,
            "thresholds": {"median": threshold(c, .5), "peak": threshold(c, .999), "steering": threshold(c, .999)},
            "peak_positions": peaks, "random_positions": random, "control_columns": controls,
            "check_columns": peers, "a": {**discovery_card, "heldout_peak_distributions": held_profile},
            "b": dict(zip(AA, b)), "c": effects,
            "d": {"later": ordered[:10], "later_share": sum(t[2] for t in ordered[:10]) / total if total else (0.0 if later else None)},
            "e": dict(zip(AA, e)),
            "checks": {"profile": pc, "write_steer": wc, "removal": {**rc["median"], "resample": rc["resample"]}}}
    dest = Path(a.out) / "cards" / key(c)
    write_json(dest.with_suffix(".json"), card)
    text = [f"# {c[0]} layer {c[1]} column {c[2]}", "", f"Insufficient: {insufficient}"]
    for section, title in zip("abcde", ("Profile", "Direct write", "Removal at peak", "Later units", "Steering")):
        text += ["", f"## {section}. {title}", "", "```json", json.dumps(card[section], indent=2), "```"]
    text += ["", "## Checks", "", "```json", json.dumps(card["checks"], indent=2), "```"]
    dest.with_suffix(".md").write_text("\n".join(text) + "\n")
    return key(c)


def columns(a):
    s = json.loads((Path(a.out) / "sample.json").read_text())
    targets = selected(s)
    Path(a.out, "cards").mkdir()
    Path(a.scratch, "direct").mkdir()
    Path(a.scratch, "direct_baselines").mkdir()
    # Parent only loads atlas/descriptor material, never a model before fork.
    G["ats"] = {p: load(Path(a.calib) / "atlases" / name) for p, name in zip(POPS,
                ("atlas_neurons_post.pt", "atlas_randdir_512_0.pt"))}
    prepare_profiles(a)
    needed = set(targets)
    for c in targets:
        needed.update(others(c, 20, 13))
    with mp.get_context("spawn").Pool(a.workers, initializer=init_worker, initargs=(vars(a),)) as pool:
        for name in pool.imap_unordered(direct, sorted(needed)):
            print("direct", name, flush=True)
        for name in pool.imap_unordered(column, targets):
            print("card", name, flush=True)


def report(a):
    cards = [json.loads(p.read_text()) for p in sorted(Path(a.out, "cards").glob("*.json"))]
    expected = selected(json.loads(Path(a.out, "sample.json").read_text()))
    if {(c["population"], c["layer"], c["column"]) for c in cards} != set(expected):
        raise ValueError("cards do not match frozen sample")

    def summary(rows):
        rates = {}
        for name in ("profile", "write_steer", "removal", "removal_resample"):
            eligible = rows if name == "write_steer" else [r for r in rows if not r["insufficient"]]
            k = sum(bool(r["checks"]["removal"]["resample"]["pass"] if name == "removal_resample"
                         else r["checks"][name]["pass"]) for r in eligible)
            n = len(eligible)
            rates[name] = {"k": k, "n": n, "rate": k / n if n else None, "ci": list(share_ci(k, n))}
        return {"n": len(rows), "insufficient": sum(r["insufficient"] for r in rows), "pass": rates}

    populations = {}
    for p in POPS:
        rows = [c for c in cards if c["population"] == p]
        populations[p] = {**summary(rows), "by_layer": {str(l): summary([r for r in rows if r["layer"] == l]) for l in range(6)}}
    result = {"populations": populations, "seeds": {"sample": SEED * 10, "heldout": 1,
              "positions": SEED, "controls": SEED + 7, "checks": SEED + 13, "resample": SEED + 19, "bootstrap": 0},
              "positions": {"peak_max": 60, "random": 20 if a.smoke else 60, "margin": 3}}
    write_json(Path(a.out) / "units.json", result)
    lines = ["# Unit study", "", "Population differences are descriptive; columns are not paired."]
    s = json.loads(Path(a.out, "sample.json").read_text())
    lines += ["", f"Direction SHA256: regenerated {s['directions_sha256']}; "
              f"calibration {s['calib_directions_sha256']}; "
              f"match = {s['directions_sha256'] == s['calib_directions_sha256']}."]
    for name in ("profile", "write_steer", "removal", "removal_resample"):
        n, d = (populations[p]["pass"][name] for p in POPS)
        diff = d["rate"] - n["rate"] if d["rate"] is not None and n["rate"] is not None else None
        lines += ["", f"{name}: neurons {n['k']}/{n['n']}, directions {d['k']}/{d['n']}; "
                  f"direction rate minus neuron rate = {diff}."]
    Path(a.out, "units.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result))


def repro_unit_writes(a):
    rows = [json.loads(line) for line in Path(a.cases).read_text().splitlines() if line.strip()]
    if not a.smoke and len(rows) != 900:
        raise ValueError("unit_writes reproduction requires all 900 cases")
    data = load(a.data)
    by_acc = dict(zip(data["acc"], data["seq"]))
    tok, _ = model_material()
    M = make_intervener()
    maxima = dict.fromkeys(("dlogp_true", "kl", "travel_kl"), 0.)
    bitwise, ok = 0, True
    last, base, ids = None, None, None
    for row in rows:
        if last != row["acc"]:
            ids = tok(by_acc[row["acc"]][:1022], return_tensors="pt")["input_ids"]
            base = M.run(ids)
            last = row["acc"]
        same = True
        i, l = row["pos"], row["layer"]
        for prefix, u in (("", row["unit"]), ("base_", row["base_unit"])):
            r = torch.nn.functional.one_hot(torch.tensor(u), 1280).float()
            edited = M.run(ids, edit=(l, r, [i], [0.]))
            kl = (base.lp_full.exp() * (base.lp_full - edited.lp_full)).sum(-1).clamp(min=0)
            other = torch.cat((kl[1:i + 1], kl[i + 2:-1]))
            got = {"dlogp_true": float(edited.lp_full[i + 1, ids[0, i + 1]] - base.lp_full[i + 1, ids[0, i + 1]]),
                   "kl": float(kl[i + 1]), "travel_kl": float(other.max()) if len(other) else 0.}
            for k, v in got.items():
                reference = row[prefix + k]
                diff = abs(v - reference)
                maxima[k] = max(maxima[k], diff)
                same &= v == reference
                ok &= diff <= 1e-6 or (k != "dlogp_true" and diff <= 1e-4 * abs(reference))
        bitwise += same
    result = {"n_cases": len(rows), "n_bitwise": bitwise, "max_abs_diff": maxima, "ok": bool(ok)}
    write_json(Path(a.out) / "repro_unit_writes.json", result)
    print(json.dumps(result))
    if not ok:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=("sample", "heldout_values", "columns", "report", "repro_unit_writes"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch")
    ap.add_argument("--calib")
    ap.add_argument("--data")
    ap.add_argument("--desc-dir")
    ap.add_argument("--desc")
    ap.add_argument("--uniref")
    ap.add_argument("--cases")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.workers < 1:
        ap.error("workers must be positive")
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    globals()[a.step](a)


if __name__ == "__main__":
    main()
