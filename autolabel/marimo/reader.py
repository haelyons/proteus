"""Read-only Janus cache adapter and the first-read journal. No inference or API calls."""

import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
JANUS = ROOT / "janus"
UNITS = ("mixed4b:431", "mixed4e:55")
CLASSES = ("cannot_tell", "one", "one_plus_related", "unrelated")
CATEGORIES = ("low (≤20)", "middle (20–80)", "high (≥80)")

# Import only pure helpers; never call their experiment/cache-building entry points.
sys.dont_write_bytecode = True
sys.path.insert(0, str(JANUS / "tools"))
import lookviews  # noqa: E402

TRANSFORMS = tuple(lookviews.battery.TRANSFORMS)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def number(value):
    return round(float(value), 4) if value is not None and math.isfinite(float(value)) else None


def category(percentile):
    if percentile is None or not math.isfinite(percentile):
        return "unavailable"
    return CATEGORIES[0] if percentile <= 20 else CATEGORIES[2] if percentile >= 80 else CATEGORIES[1]


@lru_cache(maxsize=2)
def load_packet(unit):
    if unit not in UNITS:
        raise ValueError(f"Choose one of {UNITS}, got {unit!r}")
    layer, channel = unit.split(":")
    channel = int(channel)
    sources, missing = set(), []

    def source(relative, required=False):
        path = JANUS / relative
        if not path.is_file():
            if required:
                raise FileNotFoundError(f"Required cache unavailable: {path}")
            missing.append(relative)
            return None
        sources.add(path)
        return path

    def read_json(relative, required=False):
        path = source(relative, required)
        return json.loads(path.read_text()) if path else None

    names = read_json("bank/img/names.json", True)
    labels = np.array([int(Path(name).parent.name) for name in names])
    base = f"runs/rangeview/{unit}"
    with np.load(source(f"{base}/values.npz", True), allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    if str(values["unit"]) != unit:
        raise ValueError("Unit identity differs from values cache")
    C, vC = values["C"], values["vC"]
    if len(names) != 10000 or len(C) != len(vC) or not np.isfinite(vC).all():
        raise ValueError("Unexpected corpus or activation cache")
    maximum = float(vC.max())
    rf = read_json(f"{base}/rf.json", True)
    if rf["unit"] != unit:
        raise ValueError("Unit identity differs from receptive-field cache")
    by_id = dict(zip(C.tolist(), vC.tolist()))
    top = [{"image_id": i, "raw activation": number(by_id[i]),
            "fraction of card max": number(by_id[i] / maximum),
            "hotspot row,col": f"{r},{c}", "crop box": str(box)}
           for i, r, c, box in zip(rf["top_ids"], rf["rows"], rf["cols"], rf["crop_boxes"])]
    images = {key: source(f"{base}/{key}.jpg")
              for key in ("card", "crops", "range", "nearmiss", "random")}
    distribution = [{"quantile (%)": q, "raw activation": number(np.percentile(vC, q))}
                    for q in (0, 5, 25, 50, 75, 95, 99, 100)]

    # Existing class-profile peers use all 10,000 images; Gram neighbours here use C only.
    panels = read_json(f"runs/contrastloop/{unit}/panels.json")
    peers = panels["peers"] if panels else []
    for peer in peers:
        for image_id in peer["photos"]:
            source(f"bank/img/thumbs/{image_id:05d}.jpg")
    class_names = dict(zip(panels["classes"], panels["class_names"])) if panels else {}
    means = [(int(cl), float(vC[labels[C] == cl].mean()), int((labels[C] == cl).sum()))
             for cl in np.unique(labels[C])]
    means.sort(key=lambda row: (-row[1], row[0]))
    class_rows = [{"class id": cl, "dataset label": class_names.get(cl, "label not cached"),
                   "mean raw activation (C)": number(mean), "photos (C)": n}
                  for cl, mean, n in means[:16]]
    gram = []
    layer_path = source(f"runs/proploop/values_{layer}.npy")
    if layer_path:
        V = np.load(layer_path, mmap_mode="r", allow_pickle=False)
        if V.shape != (len(names), lookviews.NCH[layer]) or V.dtype != np.float32:
            raise ValueError("Unexpected layer cache shape or dtype")
        X = np.asarray(V[C], dtype=np.float64)
        norms = np.linalg.norm(X, axis=0)
        similarities = np.divide(X.T @ X[:, channel], norms * norms[channel],
                                 out=np.full(V.shape[1], -np.inf), where=norms * norms[channel] > 0)
        similarities[channel] = -np.inf
        gram = [{"unit": f"{layer}:{int(c)}", "activation cosine (C)": number(similarities[c])}
                for c in np.argsort(-similarities, kind="stable")[:3] if np.isfinite(similarities[c])]

    profile = []
    profile_path = source(f"runs/lookviews/profile_{layer}.npz")
    if profile_path:
        with np.load(profile_path, allow_pickle=False) as archive:
            d = {key: archive[key] for key in archive.files}
        if str(d["layer"]) != layer or tuple(d["transforms"]) != TRANSFORMS:
            raise ValueError("Profile identity or transform order mismatch")
        if len(d["P0"]) != 1000 or not set(d["P0"].tolist()) <= set(C.tolist()):
            raise ValueError("Profile pool is not the expected subset of C")
        _, rows = lookviews.unit_profile(lookviews.retention_table(d), channel)
        for transform, retention, median, percentile in rows:
            profile.append({"transform": transform, "retention": float(retention),
                            "layer median": float(median), "layer percentile": float(percentile),
                            "photos": min(20, int((d["orig"][:, channel] > 0).sum()))})

    strata = []
    csv_path = source(f"runs/rangeprofile/{layer}.csv")
    if csv_path:
        with csv_path.open() as stream:
            all_rows = list(csv.DictReader(stream))
        selected = next(row for row in all_rows if row["unit"] == unit)
        for transform in TRANSFORMS:
            for s, bounds in enumerate(("1–20", "21–100", "101–300"), 1):
                key = f"{transform}_ret_s{s}"
                baselines = [float(row[key]) for row in all_rows if row[key]]
                strata.append({"transform": transform, "positive ranks in P0": bounds,
                               "retention": number(selected[key] or None),
                               "layer median": number(np.median(baselines)) if baselines else None,
                               "retention percentile": number(selected[f"{transform}_ret_pct_s{s}"] or None),
                               "change / P0 max": number(selected[f"{transform}_chg_s{s}"] or None),
                               "change percentile": number(selected[f"{transform}_chg_pct_s{s}"] or None)})
    spared = read_json(f"runs/sparedview/{unit}/spared.json")
    for key in ("affected", "spared"):
        images[key] = source(f"runs/sparedview/{unit}/{key}.jpg")
    pair_stats = read_json(f"runs/pairtable/{unit}/stats.json")
    pair_rows = []
    if pair_stats:
        for row in pair_stats["picks"]["every"]:
            out = {"row": row["row"], "question": row["question"],
                   "higher on D": row["first_answer"], "lower on D": row["second_answer"]}
            for split in ("D", "T", "T_null"):
                item = row[split]
                out.update({f"{split} n (higher/lower)": f"{item['n_first']}/{item['n_second']}",
                            f"{split} d": number(item["d"]),
                            f"{split} percentile": number(item["pct"]),
                            f"{split} peer percentiles": str([number(x) for x in item["peer_pct"]])
                            if item["peer_pct"] else "unavailable"})
                if split != "D":
                    out[f"{split} outcome"] = item["outcome"]
                    out[f"{split} distinctive"] = item["distinctive"]
            pair_rows.append(out)
    basis_path = source("runs/r32/report.txt")
    basis = []
    if basis_path:
        lines = basis_path.read_text().splitlines()
        keys = lines[0].split()
        basis = [dict(zip(keys, line.split())) for line in lines[1:]
                 if line.startswith(layer + " ") and line.split()[1] in ("channel", "direction")]

    wiring, wiring_note = [], "No cached input/contribution/reader view for this unit. Downstream use is not measured."
    if unit == "mixed4e:55":
        images["negative"] = source("bank/cards/U-860_neg.jpg")
        images["wiring"] = source("runs/featvis/renders.jpg")
        circuit_path = source("runs/circuit/circuit.npz")
        map_path = source("bank/img/acts.npz")
        if circuit_path and map_path:
            with np.load(map_path, allow_pickle=False) as archive:
                pre = archive["maps"].astype(np.float32)
            positive = np.maximum(pre.max(axis=(1, 2)), 0)
            firing = positive >= 0.3 * positive.max()
            with np.load(circuit_path, allow_pickle=False) as archive:
                w = archive["w"]
                drive = archive["contrib"][firing].astype(np.float64).mean(axis=0)
                bias = float(archive["b"])
            weight_order = np.argsort(-np.abs(w), kind="stable")
            drive_order = np.argsort(-np.abs(drive), kind="stable")
            weight_rank, drive_rank = np.argsort(weight_order) + 1, np.argsort(drive_order) + 1
            selected = list(dict.fromkeys([*weight_order[:12], *drive_order[:6]]))
            wiring = [{"input": f"mixed4d:{int(c)}", "signed weight": number(w[c]),
                       "rank |weight|": int(weight_rank[c]), "mean signed contribution": number(drive[c]),
                       "rank |mean contribution|": int(drive_rank[c])} for c in selected]
            wiring_note = (f"Inputs to mixed4e_1x1_pre_relu_conv[55]; bias {bias:.4f}. "
                           f"Mean w×activation at the target hotspot over {int(firing.sum())} original-crop photos "
                           "with spatial maximum ≥30% of the 10,000-photo corpus maximum. "
                           "Weights are float32; cached contributions and signed maps are float16. "
                           "Reader renders cover only three positive mixed5a 1×1 connections after pooling. "
                           "Outgoing suppressors, other branches and downstream causal use are not measured here.")

    # Include methods and references shown by the notebook, not just numeric caches.
    for relative in ("tools/lookviews.py", "tools/battery.py", "tools/rangeprofile.py", "tools/contrastloop.py",
                     "tools/proploop.py", "tools/pairtable.py", "tools/circuit.py", "tools/featvis.py",
                     "bank/img_acts.py", "bank/img_cards_v2.py", "README.md", "CONTRACT.md"):
        source(relative)
    sources.update([Path(__file__).resolve(), HERE / "first_read.py"])
    manifest = {str(path.relative_to(ROOT)): {"sha256": digest(path), "bytes": path.stat().st_size}
                for path in sorted(sources)}
    return {"unit": unit, "layer": layer, "channel": channel, "n_card": len(C), "maximum": maximum,
            "zero_fraction": float(np.mean(vC == 0)), "rf_side": rf["S"], "top": top,
            "distribution": distribution, "class_means": class_rows, "images": images,
            "peers": peers, "gram": gram, "profile": profile, "strata": strata, "spared": spared,
            "pair_rows": pair_rows, "pair_counts": (len(pair_stats["D"]), len(pair_stats["T"])) if pair_stats else None,
            "basis": basis, "wiring": wiring, "wiring_note": wiring_note, "manifest": manifest,
            "missing": missing}


def form_fields(fields):
    """marimo form validation receives dropdowns as lists; callbacks receive scalars."""
    return {key: value[0] if isinstance(value, list) and len(value) == 1 else value
            for key, value in fields.items()} if fields else fields


def validate_prediction(fields):
    fields = form_fields(fields)
    if not fields:
        return "Complete the prediction before revealing the measurements."
    for key in ("reader", "prior_exposure", "response", "evidence", "rival"):
        if not isinstance(fields.get(key), str) or not fields[key].strip():
            return f"Please fill in {key.replace('_', ' ')}."
    if fields.get("transform") not in TRANSFORMS or fields.get("category") not in CATEGORIES:
        return "Choose a transformation and a layer-relative category."
    low, high = fields.get("low"), fields.get("high")
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in (low, high)):
        return "Both retention bounds must be finite numbers."
    if not 0 <= low <= high:
        return "Retention bounds must satisfy 0 ≤ lower ≤ upper. Enhancement can exceed 1."
    confidence = fields.get("confidence")
    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return "Confidence must be between 0 and 1."
    return None


def validate_reading(fields):
    fields = form_fields(fields)
    if not fields:
        return "Complete the revised reading."
    for key in ("response", "wiring", "family", "failures", "missing_evidence", "next_test", "evidence"):
        if not isinstance(fields.get(key), str) or not fields[key].strip():
            return f"Please fill in {key.replace('_', ' ')}; 'not measured' is valid."
    if fields.get("class") not in CLASSES:
        return "Choose a provisional class."
    confidence = fields.get("confidence")
    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return "Confidence must be between 0 and 1."
    return None


def score_prediction(packet, fields):
    row = next((row for row in packet["profile"] if row["transform"] == fields["transform"]), None)
    if row is None or not all(math.isfinite(row[key]) for key in ("retention", "layer percentile")):
        return {"status": "unavailable", "interval_hit": None, "category_hit": None}
    return {"status": "cached replay", "measurement": row,
            "interval_hit": fields["low"] <= row["retention"] <= fields["high"],
            "category_hit": category(row["layer percentile"]) == fields["category"],
            "measured_category": category(row["layer percentile"])}


class ReadSession:
    """One immutable prediction, outcome and revision, bound to one cached packet."""

    def __init__(self, packet, directory=None):
        self.packet = packet
        self.directory = Path(directory or HERE / "readings").resolve()
        if not self.directory.is_relative_to(HERE):
            raise ValueError("Journal writes must stay under autolabel/marimo")
        self.id = uuid4().hex
        self.prediction = self.outcome = self.reading = None

    def _save(self, kind, content):
        self.directory.mkdir(parents=True, exist_ok=True)
        record = {"schema": 1, "session": self.id, "kind": kind, "unit": self.packet["unit"],
                  "utc": datetime.now(timezone.utc).isoformat(), **content}
        data = json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
        path = self.directory / f"{self.id}.{kind}.json"
        with path.open("x", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return {"path": str(path), "sha256": hashlib.sha256(data.encode()).hexdigest(), "record": record}

    def commit_and_reveal(self, fields):
        error = validate_prediction(fields)
        if error:
            raise ValueError(error)
        if self.prediction is None:
            for relative, fingerprint in self.packet["manifest"].items():
                if digest(ROOT / relative) != fingerprint["sha256"]:
                    raise ValueError(f"Source changed; reload the notebook before committing: {relative}")
            self.prediction = self._save("prediction", {
                "fields": dict(fields), "sources": self.packet["manifest"], "missing": self.packet["missing"],
                "protocol": "prediction-before-reveal replay; not blind validation",
                "visible_before_commit": ["identity", "top/crops", "range", "distribution", "negative", "peers", "wiring"],
                "rule": {"statistic": "median transformed/original on positive top 20 of P0 (1000 card photos)",
                         "interval": "inclusive", "category": "low ≤20; middle >20 and <80; high ≥80",
                         "semantic_acceptance": "none"}})
        elif fields != self.prediction["record"]["fields"]:
            raise ValueError("This session's prediction is locked. Start a new session to change it.")
        if self.outcome is None:
            self.outcome = self._save("outcome", {"prediction_sha256": self.prediction["sha256"],
                                                 **score_prediction(self.packet, fields)})
        return {"prediction": self.prediction, "outcome": self.outcome}

    def save_reading(self, fields):
        error = validate_reading(fields)
        if error:
            raise ValueError(error)
        if self.outcome is None:
            raise ValueError("Commit and reveal before saving the revised reading.")
        if self.reading is None:
            self.reading = self._save("reading", {"prediction_sha256": self.prediction["sha256"],
                                                 "outcome_sha256": self.outcome["sha256"], "fields": dict(fields)})
        elif fields != self.reading["record"]["fields"]:
            raise ValueError("This session's reading is already saved.")
        return self.reading
