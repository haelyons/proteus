"""Column sources sharing ESM-2 pre- and post-GELU hook points."""
from __future__ import annotations

import os

import torch

from .core import _mm_f32out


class Source:
    def __init__(self, name: str, hook: str, width: int, d_ff: int, seed=None):
        self.name, self.hook, self.width = name, hook, width
        self.d_ff, self.seed = d_ff, seed
        self._directions = {}

    def directions(self, l: int, device, dtype) -> torch.Tensor | None:
        if not self.name.startswith("randdir"):
            return None
        key = (l, torch.device(device), dtype)
        if key not in self._directions:
            cpu_key = (l, torch.device("cpu"), torch.float32)
            if cpu_key not in self._directions:
                if self.name == "randdir_identity":
                    R = torch.eye(self.d_ff, dtype=torch.float32)
                else:
                    g = torch.Generator().manual_seed(self.seed * 1000 + l)
                    R = torch.randn(self.d_ff, self.width, generator=g, dtype=torch.float32)
                    R /= R.norm(dim=0)
                self._directions[cpu_key] = R
            self._directions[key] = self._directions[cpu_key].to(device=device, dtype=dtype)
        return self._directions[key]

    def __call__(self, l: int, H: torch.Tensor) -> torch.Tensor:
        R = self.directions(l, H.device, H.dtype)
        return H if R is None else _mm_f32out(H, R)


def parse_sources(spec: str, d_ff: int) -> list[Source]:
    sources = []
    for item in spec.split(","):
        if item in ("neurons_pre", "neurons_post"):
            source = Source(item, "pre" if item == "neurons_pre" else "post", d_ff, d_ff)
        elif item == "randdir_identity" and os.environ.get("PROTEUS_TEST_SOURCES") == "1":
            source = Source(item, "post", d_ff, d_ff)
        elif item.startswith("randdir:"):
            try:
                _, m, seed = item.split(":")
                m, seed = int(m), int(seed)
                if m <= 0:
                    raise ValueError
                torch.Generator().manual_seed(seed * 1000)
            except (ValueError, RuntimeError, OverflowError):
                raise ValueError(f"invalid column source: {item}") from None
            source = Source(f"randdir_{m}_{seed}", "post", m, d_ff, seed)
        else:
            raise ValueError(f"invalid column source: {item}")
        if any(s.name == source.name for s in sources):
            raise ValueError(f"duplicate column source: {source.name}")
        sources.append(source)
    return sources


def attach_sources(model, sources, sink, *, wants=None):
    """Call sink(source_index, layer, X) only when wants(source_index), if supplied."""
    handles = []
    for l, layer in enumerate(model.encoder.layer):
        for point, mod in (("pre", layer.intermediate.dense), ("post", layer.intermediate)):
            readers = [(i, s) for i, s in enumerate(sources) if s.hook == point]
            if not readers:
                continue

            def hook(_m, _inp, out, l=l, readers=readers):
                H = out.reshape(-1, out.shape[-1])
                for i, source in readers:
                    if wants is None or wants(i):
                        sink(i, l, source(l, H))

            handles.append(mod.register_forward_hook(hook))
    return handles
