"""Timing and memory pilot for the 650M L15/2087 use study (results/ubq650_use_plan.md). Times only: no model output
is printed, saved or inspected. ESM-2 650M EsmForMaskedLM, fp32, CPU.

Measures, for constructs of 2 and 4 blocks of 80 residues (batch 1 and 4) and 8 blocks (batch 1):
  plain   : masked-LM forward (log-softmax at 2 positions, discarded)
  capture : + hooks gathering pre-GELU values of ~170 named units across all 33 layers, and the full layer-15
            post-GELU tensor (needed to choose write-matched controls)
  patch   : + a layer-15 post-GELU hook replacing unit 2087 (and a 20-unit group) at every position from a donor
            tensor, with graded interpolation
Also: model load time and peak resident memory.

  python scripts/ubq650_pilot_timing.py --data data/swissprot_2026_03.pt --out pilot.json
"""
from __future__ import annotations

import argparse
import json
import random
import resource
import statistics as st
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6      # Linux: KB -> GB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--held", default="lists/heldout_created_after_2021-11-17.tsv"); ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    torch.set_grad_enabled(False); torch.set_num_threads(4)
    from transformers import EsmForMaskedLM, EsmTokenizer
    res = {"threads": torch.get_num_threads(), "rss_before_load_gb": round(rss_gb(), 2)}
    t0 = time.time()
    hf = "facebook/esm2_t33_650M_UR50D"; tok = EsmTokenizer.from_pretrained(hf); m = EsmForMaskedLM.from_pretrained(hf).float().eval()
    res["load_s"] = round(time.time() - t0, 1); res["rss_after_load_gb"] = round(rss_gb(), 2)
    # a discovery (non-held-out) fragment as block content; outputs are never looked at
    d = torch.load(a.data, weights_only=False)
    held = {x.split("\t")[0] for x in open(a.held) if not x.startswith("Entry")}
    rng = random.Random(0)
    pool = [i for i, x in enumerate(d["acc"]) if x not in held and len(d["seq"][i]) >= 200]
    s = d["seq"][rng.choice(pool)]; block = s[50:130]
    layers = m.esm.encoder.layer
    units = {l: sorted(rng.sample(range(5120), 5)) for l in range(33)}; units[15] = sorted(set(units[15]) | {2087})
    state = {"capture": False, "patch": None, "store": {}}

    def mk(l):
        def pre(_m, _i, o):
            if state["capture"]:
                state["store"][("pre", l)] = o[:, :, units[l]].clone()
        return pre

    def post15(_m, _i, o):
        if state["capture"]:
            state["store"]["post15"] = o.clone()
        if state["patch"] is not None:
            idx, donor, alpha = state["patch"]
            o = o.clone(); o[:, :, idx] = (1 - alpha) * o[:, :, idx] + alpha * donor[:, :, idx]
        return o
    for l in range(33):
        layers[l].intermediate.dense.register_forward_hook(mk(l))
    layers[15].intermediate.register_forward_hook(post15)
    group = sorted(set(rng.sample(range(5120), 19)) | {2087})

    def run(ids):
        out = m(input_ids=ids).logits
        return torch.log_softmax(out[:, :3].double(), -1)                  # computed, discarded

    timing = []
    for nb, bss in ((2, (1, 4)), (4, (1, 4)), (8, (1,))):
        seq = block * nb
        ids1 = tok(seq, return_tensors="pt")["input_ids"]; ids1[0, 2] = tok.mask_token_id
        for bs in bss:
            ids = ids1.repeat(bs, 1)
            row = {"blocks": nb, "length": len(seq), "batch": bs}
            for mode in ("plain", "capture", "patch_unit", "patch_group"):
                state["capture"] = mode == "capture"; state["patch"] = None
                if mode.startswith("patch"):
                    donor = torch.zeros(bs, ids.shape[1], 5120)
                    state["patch"] = (torch.tensor([2087]) if mode == "patch_unit" else torch.tensor(group), donor, 0.5)
                run(ids)                                                    # warm-up
                ts = []
                for _ in range(a.reps):
                    t = time.time(); run(ids); ts.append(time.time() - t)
                row[mode + "_s_per_forward_batch"] = round(st.median(ts), 3)
                row[mode + "_s_per_sequence"] = round(st.median(ts) / bs, 3)
                state["store"].clear()
            row["peak_rss_gb"] = round(rss_gb(), 2)
            timing.append(row); print(json.dumps(row), flush=True)
    res["timing"] = timing; res["peak_rss_gb"] = round(rss_gb(), 2)
    res["note"] = "times only; logits computed and discarded, never printed or saved"
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
