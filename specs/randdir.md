# Spec: column sources in the atlas run: post-GELU neurons and random directions (unit 5b)

Status: frozen at the commit that adds this file. Replaces section 5b of `specs/columns.md`.
Without `--columns`, `scripts/run_atlas.py` behaves and writes exactly as before (bitwise).

## Why

The basis control needs random directions in the post-GELU MLP space, collected by the same
atlas. Every column source except shuffled weights reads the same forward pass, and the forward
is about 80% of a batch's cost, so several sources share one forward.

## Sources

| Spec | Name | Hook | Columns per layer | Values handed to the Atlas |
| --- | --- | --- | --- | --- |
| `neurons_pre` (default) | `neurons_pre` | output of `layer.intermediate.dense` (today's hook) | d_ff | as today |
| `neurons_post` | `neurons_post` | output of `layer.intermediate` (after the model's own GELU; what the next layer reads) | d_ff | as the model computes them, model dtype |
| `randdir:M:SEED` | `randdir_M_SEED` | post-GELU, as above | M | `atlas.core._mm_f32out(H, R_l)`: fp32 |

`R_l` (`[d_ff, M]`, layer l): `torch.randn(d_ff, M, generator=torch.Generator().manual_seed(SEED
* 1000 + l), dtype=torch.float32)`, each column divided by its L2 norm, then cast once to the
model dtype on the device. A test-only spec `randdir_identity` (accepted only when the environment
variable `PROTEUS_TEST_SOURCES=1` is set) uses `R_l = I` (M = d_ff) and is named
`randdir_identity`.

## `atlas/columns.py`

```
parse_sources(spec: str, d_ff: int) -> list[Source]          # "neurons_pre,randdir:512:0" -> sources, in that order; duplicates are an error
class Source:
    name: str
    hook: str                     # "pre" | "post"
    width: int
    def directions(self, l: int, device, dtype) -> Tensor | None   # R_l, or None for neurons
    def __call__(self, l: int, H: Tensor) -> Tensor              # [T, d] -> [T, width]
attach_sources(model, sources, sink) -> list[handle]             # sink(source_index, l, X); one hook per hook point; each hook point's tensor is computed once per forward
```

## `scripts/run_atlas.py`

New argument `--columns SPEC[,SPEC...]` (default `neurons_pre`). With one source and that source
`neurons_pre`, every output is as before. Otherwise:

- One Atlas per source, in spec order, each with its own pilot edges and seeds computed from the
  same pilot forward (the pilot collects every source's values for its pilot rows).
- Overflow redo: after a batch every Atlas reports its redo layers. If any is non-empty, one more
  forward runs in which **only** the Atlases with a non-empty list take part (`begin_batch(ctx,
  redo_layers=...)`, observe, `end_batch`), and the hook dispatch skips every other Atlas for that
  forward. An Atlas that did not take part is not touched (its `n_seen` and statistics are
  counted once).
- Outputs, per source name N: `atlas_N.pt` (or `shard_N.pt` with `--shard`, which also records
  `"source": N`); for `randdir` sources `directions_N.pt` = `{"seed", "M", "R": [R_0, ...]
  (fp32, CPU), "sha256"}` where `sha256` is over the concatenated bytes of the fp32 `R_l` in layer
  order.
- `run.json` gains `"columns": SPEC` and `"sources": {N: {"width", "digests" (unsharded),
  "stats", "n_seen", "res_seeded_bins", "res_seed_limited_bins", "top_seed_short_groups",
  "directions_sha256" (randdir only)}}`; the top-level `stats`, `digests`, `n_seen` refer to the
  first source.
- `batches.jsonl` records gain `"redo_layers_by_source": {N: [...]}`; `redo_layers` is their
  union.
- Pilot artifact with several sources: `{"sources": {N: {"edges", "top_seed", "res_seed_key"}},
  "n_total", "n_batches", "plan_sha256", "args_key", "model"}` with `args_key` including
  `columns`. With one `neurons_pre` source the artifact is exactly as before.
- `scripts/merge_atlas.py` refuses inputs whose `source` differs (absent counts as
  `neurons_pre`). Merging is per source: pass that source's `shard_N.pt` files.

## Worked example

`--model 8M --columns neurons_pre,randdir:512:0` (8M: 6 layers, d_ff 1280) writes
`atlas_neurons_pre.pt` (`top_val` shape `[6, 2, 1280, K]`), `atlas_randdir_512_0.pt` (`top_val`
`[6, 2, 512, K]`) and `directions_randdir_512_0.pt` with `R[3]` generated from seed `0 * 1000 + 3
= 3`; `run.json["sources"]` has the keys `neurons_pre` and `randdir_512_0`.

## Acceptance (blind tests; ESM-2 8M, CPU, fp32, one thread)

A. Default and `--columns neurons_pre`: `atlas.pt` bitwise equal to the code at the spec commit
   (exact fields and float sums), same digests.
B. `--columns neurons_pre,neurons_post`: `atlas_neurons_pre.pt` bitwise equal to A's `atlas.pt`
   (sharing a forward changes nothing, redo included); for at least 99% of the units whose
   pre-GELU top-K values (positive pole) are all > 0, the post-GELU positive-pole `top_tok` equals
   the pre-GELU one (GELU is increasing there).
C. `randdir:64:0`: two runs bitwise equal; `--shard 0/2` and `1/2` from one pilot artifact, merged,
   equal the unsharded run (exact fields bitwise, float sums within 1e-9 with
   `--gram-flush-every 1`).
D. With `PROTEUS_TEST_SOURCES=1`, `--columns neurons_post,randdir_identity`: the two atlases are
   equal (exact fields bitwise; float sums within 1e-9).
E. `directions_*.pt`: `R_l` columns have unit norm (within 1e-6); `sha256` recomputes; changing
   SEED changes every `R_l`.
F. In every multi-source run of A-D, each source's `run.json` `n_seen` equals the run's
   `n_total_plan` (the valid tokens streamed), whatever the per-source redo; the number of batches
   whose `redo_layers_by_source` differ between sources is reported by the test (not asserted).
