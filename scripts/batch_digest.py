"""Hash native pre-GELU FFN activations for selected global batches, or compare hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.data import SwissProt, plan_digest, shuffled_order  # noqa: E402
from atlas.esm import attach, load  # noqa: E402


def compare(paths):
    a, b = [json.loads(Path(p).read_text()) for p in paths]
    if a["plan_sha256"] != b["plan_sha256"]:
        print("plan_sha256 differs")
        return 1
    for batch in sorted(a["batches"].keys() | b["batches"].keys(), key=int):
        x, y = a["batches"].get(batch), b["batches"].get(batch)
        if x is None or y is None:
            print(f"batch {batch}, layer 0: batch missing")
            return 1
        if x["shape"] != y["shape"]:
            print(f"batch {batch}, layer 0: shape differs")
            return 1
        for layer in range(max(len(x["layers"]), len(y["layers"]))):
            if (layer >= len(x["layers"]) or layer >= len(y["layers"])
                    or x["layers"][layer] != y["layers"][layer]):
                print(f"batch {batch}, layer {layer}: digest differs")
                return 1
    print("all batch/layer digests match")
    return 0


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compare", nargs=2, metavar=("J1", "J2"))
    ap.add_argument("--model", default="8M")
    ap.add_argument("--data")
    ap.add_argument("--batch", type=int, nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens-per-batch", type=int, default=32768)
    ap.add_argument("--window", type=int, default=8192)
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--exclude-acc")
    ap.add_argument("--max-batches", type=int, default=0)
    args = ap.parse_args()
    if args.compare:
        return compare(args.compare)
    if not args.data or args.batch is None or not args.out:
        ap.error("--data, --batch and --out are required unless --compare is used")
    ds = SwissProt(args.data)
    order = shuffled_order(len(ds), args.seed)
    if args.exclude_acc:
        excluded = {ln.split("\t")[0].split(",")[0].strip() for ln in open(args.exclude_acc)
                    if ln.strip() and not ln.startswith("Entry")}
        drop = torch.tensor([acc in excluded for acc in ds.d["acc"]], dtype=torch.bool)
        order = order[~drop[order]]
    if args.max_seqs:
        order = order[:args.max_seqs]
    plan = ds.plan(order, args.tokens_per_batch, args.window, args.seed)
    if args.max_batches:
        plan = plan[:args.max_batches]
    if any(b < 0 or b >= len(plan) for b in args.batch):
        ap.error(f"--batch indices must be in 0..{len(plan) - 1}")
    dev = torch.device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    model, tok, info = load(args.model, dtype=dtype, device=dev, seed=args.seed)
    device = {"type": dev.type, "name": platform.processor() or platform.machine(), "uuid": None}
    if dev.type == "cuda":
        props = torch.cuda.get_device_properties(dev)
        device.update(name=props.name, uuid=str(props.uuid) if hasattr(props, "uuid") else None)
    report = {"plan_sha256": plan_digest(plan), "device": device,
              "versions": {"torch": torch.__version__, "cuda": torch.version.cuda}, "batches": {}}
    layers = []

    def sink(layer, h):
        # uint8 view preserves native bytes, including bf16 (unsupported by NumPy).
        layers[layer] = hashlib.sha256(h.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()

    handles = attach(model, sink)
    try:
        for bi in sorted(set(args.batch)):
            batch = next(ds.batches(tok.get_vocab(), order, args.tokens_per_batch, plan=[plan[bi]]))
            layers = [None] * info["n_layers"]
            model(input_ids=batch.input_ids.to(dev), attention_mask=batch.attention_mask.to(dev))
            report["batches"][str(bi)] = {"shape": list(batch.input_ids.shape), "layers": layers}
    finally:
        for handle in handles:
            handle.remove()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
