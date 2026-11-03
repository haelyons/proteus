"""Summarise the A10 profiling job: overhead decay plot, power/clock per batch, forward profile.

  python scripts/analyze_a10.py <job_dir> [--out <dir>]
job_dir holds profile_forward.json, smi.csv (optional), seeded/ and long/ (batches.jsonl, run.json).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics as st
from pathlib import Path


def load_jsonl(p: Path):
    if not p.exists():
        return []
    seen, out = set(), []
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r["batch"] in seen:  # replayed after a resume: keep the last
            out = [x for x in out if x["batch"] != r["batch"]]
        seen.add(r["batch"])
        out.append(r)
    return sorted(out, key=lambda r: r["batch"])


def load_smi(p: Path):
    if not p.exists():
        return [], []
    rows = list(csv.reader(p.read_text().splitlines()))
    hdr = [h.strip() for h in rows[0]]
    out = []
    for r in rows[1:]:
        if len(r) != len(hdr):
            continue
        d = dict(zip(hdr, (x.strip() for x in r)))
        try:
            t = dt.datetime.strptime(d["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
        except ValueError:
            continue
        rec = {"t": t}
        for k, v in d.items():
            if k == "timestamp":
                continue
            try:
                rec[k] = float(v)
            except ValueError:
                rec[k] = v
        out.append(rec)
    return hdr, out


def window(smi, t0, t1, key):
    v = [s[key] for s in smi if t0 <= s["t"] <= t1 and isinstance(s.get(key), float)]
    return st.mean(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    J = Path(a.job_dir)
    out = Path(a.out or J)
    out.mkdir(parents=True, exist_ok=True)
    S = {}
    hdr, smi = load_smi(J / "smi.csv")
    pw = next((h for h in hdr if h.startswith("power.draw")), None)
    sm = next((h for h in hdr if h.startswith("clocks.sm")), None)
    cap = next((h for h in hdr if "sw_power_cap" in h), None)
    pf = json.loads((J / "profile_forward.json").read_text()) if (J / "profile_forward.json").exists() else {}
    S["forward_profile"] = {k: pf.get(k) for k in ("env", "determinism", "gemm")}
    if "profile" in pf:
        p = pf["profile"]
        S["forward_profile"]["class_share"] = p.get("class_share")
        S["forward_profile"]["device_ms_per_batch_by_class"] = p.get("device_ms_per_batch_by_class")
        S["forward_profile"]["gemm_tflops_on_gemm_time"] = p.get("gemm_tflops_on_gemm_time")
        S["forward_profile"]["top_kernels"] = p.get("top_kernels", [])[:12]
    if smi and pf.get("gemm"):
        S["gemm_power"] = {k: {"tflops": v["tflops"], "power_W": window(smi, v["t_start"], v["t_end"], pw),
                               "sm_MHz": window(smi, v["t_start"], v["t_end"], sm)} for k, v in pf["gemm"].items()}
    if smi and pf.get("forward"):
        fb = pf["forward"]["batches"]
        S["forward_power"] = {"power_W": st.mean(filter(None, (window(smi, r["t_start"], r["t_end"], pw) for r in fb))) if fb else None,
                              "sm_MHz": st.mean(filter(None, (window(smi, r["t_start"], r["t_end"], sm) for r in fb))) if fb else None,
                              "power_cap_active_frac": window(smi, fb[0]["t_start"], fb[-1]["t_end"], cap) if cap and fb else None}
    S["tiles"] = pf.get("tiles")
    runs = {}
    for name in ("seeded", "long"):
        R = load_jsonl(J / name / "batches.jsonl")
        if not R:
            continue
        info = json.loads((J / name / "run.json").read_text()) if (J / name / "run.json").exists() else {}
        ov = [r["overhead_vs_fwd"] for r in R if r.get("overhead_vs_fwd") is not None]
        segs = {}
        for lo, hi in [(0, 10), (10, 50), (50, 100), (100, 200), (200, 400), (400, 620)]:
            x = [r["overhead_vs_fwd"] for r in R if lo <= r["batch"] < hi and r.get("overhead_vs_fwd") is not None]
            if x:
                segs[f"{lo}-{hi}"] = {"median": st.median(x), "mean": st.mean(x), "n": len(x),
                                      "redo_batches": sum(1 for r in R if lo <= r["batch"] < hi and r["redo_layers"]),
                                      "compacted_frac_median": st.median(r["compacted_frac"] for r in R if lo <= r["batch"] < hi)}
        tot_fwd = sum(r["fwd_only_s"] for r in R if r.get("fwd_only_s"))
        tot_atl = sum(r["atlas_pass_s"] for r in R if r.get("atlas_pass_s"))
        runs[name] = {"batches": len(R), "tokens": R[-1]["n_seen_before"] + R[-1]["n_valid"],
                      "overhead_by_segment": segs, "overhead_pooled": tot_atl / tot_fwd - 1 if tot_fwd else None,
                      "ckpt_s": [r["ckpt_s"] for r in R if r.get("ckpt_s")],
                      "final_save_s": info.get("final_save_s"), "atlas_pt_GB": info.get("atlas_pt_GB"),
                      "seed_verification": {k: info.get(k) for k in ("res_seeded_bins", "res_seed_limited_bins", "top_seed_short_groups")},
                      "fwd_power_W": st.mean(filter(None, (window(smi, r["t_start"], r["t_start"] + r["fwd_only_s"], pw) for r in R if r.get("fwd_only_s")))) if smi and pw else None,
                      "fwd_sm_MHz": st.mean(filter(None, (window(smi, r["t_start"], r["t_start"] + r["fwd_only_s"], sm) for r in R if r.get("fwd_only_s")))) if smi and sm else None}
        runs[name]["_series"] = [(r["batch"], r.get("overhead_vs_fwd"), bool(r["redo_layers"]), r["n_seen_before"]) for r in R]
    S["runs"] = {k: {kk: vv for kk, vv in v.items() if kk != "_series"} for k, v in runs.items()}
    (out / "a10_summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1)[:6000])
    # ---- decay plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
        cols = {"long": ("#2a78d6", "unseeded (current design)"), "seeded": ("#eb6834", "seeded thresholds")}
        plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})
        fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURF)
        ax.set_facecolor(SURF)
        for name, (c, lab) in cols.items():
            if name not in runs:
                continue
            ser = [(b, o, rd) for b, o, rd, _ in runs[name]["_series"] if o is not None]
            xs = [b for b, _, _ in ser]
            ys = [max(o, 1e-3) for _, o, _ in ser]
            ax.scatter([b for b, _, rd in ser if not rd], [max(o, 1e-3) for _, o, rd in ser if not rd], s=8, color=c, alpha=0.35, linewidths=0)
            ax.scatter([b for b, _, rd in ser if rd], [max(o, 1e-3) for _, o, rd in ser if rd], s=22, color=c, marker="x", linewidths=1)
            w = 15
            med = [st.median(ys[max(0, i - w // 2): i + w // 2 + 1]) for i in range(len(ys))]
            ax.plot(xs, med, color=c, lw=2, label=lab)
            ax.annotate(lab, (xs[-1], med[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
        ax.set_yscale("log")
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.set_xlabel("batch index (~32k rows each)")
        ax.set_ylabel("atlas overhead vs paired forward (log)")
        ax.set_title("Whole-atlas overhead per batch, ESM-2 650M on one A10 (dots: batches; x: dense redo; line: rolling median of 15)",
                     loc="left", color=INK, fontsize=10.5)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
        fig.tight_layout()
        fig.savefig(out / "overhead_decay.png", dpi=200, facecolor=SURF)
        print("plot ->", out / "overhead_decay.png")
    except Exception as ex:  # noqa: BLE001
        print("plot failed:", ex)


if __name__ == "__main__":
    main()
