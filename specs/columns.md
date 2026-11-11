# Spec: column sources (unit 5)

Status: draft for review; frozen at the commit that marks it frozen. Order is fixed: 5a, then 5b,
then 5c. No real-source result is read before 5a's gate has been run and recorded.

## Why

The science question compares the share of "structural" calls across column sources run through
one atlas and one decision rule (ruler v1.2, unchanged). The atlas is rank-based, so the ruler
was validated on Gaussian planted columns on the argument that marginal shape does not matter.
That argument holds for continuous columns only. The new sources differ in shape:

| Source | Marginal shape | What can break |
| --- | --- | --- |
| Neurons, post-GELU | floor at about -0.17, long positive tail | negative pole is an off state; ties near the floor |
| Random directions in post-GELU space | sum of thousands of GELU outputs: continuous, two-sided, mildly skewed | none expected; the control case |
| SAE latents (InterPLM) | exact zero for most tokens, one pole | quantile edges collapse onto 0; the negative pole is all ties; the peak bin can be empty when the firing rate is below 1e-5 |

So the ruler is tested on planted columns of each shape first (5a).

## 5a. Planted columns of each shape

`scripts/synthetic_columns.py` gains `--shape {gauss, gelu, gelu_off, randdir, sae}` (default
`gauss`, unchanged output) and `--sae-rate R` (default 0.01). With `z = alpha * signal + noise`
as today, the stored column is:

| Shape | Column | Notes |
| --- | --- | --- |
| `gauss` | `z` | today's design |
| `gelu` | `gelu(z)` | neuron on its typical operating point |
| `gelu_off` | `gelu(z - 2)` | neuron mostly in its off state |
| `randdir` | `(alpha * signal + sum_j w_j * gelu(n_j)) / s` with 64 independent `n_j ~ N(0,1)`, `w` a seeded unit vector, `s` the noise-only standard deviation | the signal is planted at the same SNR as `gauss`; only the noise shape changes |
| `sae` | `relu(z - t)` with `t` the `1 - R` quantile of `N(0,1)` | firing rate `R` on noise-only columns; exact zeros otherwise |

Every other part of the pipeline (pilot edges, atlas, `run_ruler.py`, `eval_synthetic.py`) is
unchanged, and `eval_synthetic.py` grades each shape against the same `EXPECT` table.

Measurement (CPU; needs the descriptor subset on the Lambda filesystem, see `INVENTORY.md`):
shapes `gauss gelu gelu_off randdir sae` with `--sae-rate 0.1, 0.01, 0.001`, three noise draws
each, same proteins and seeds as `results/synthetic_ruler_20261105`. Predictions are written and
committed before the run. Gate for using a source's shape: on that shape, (i) each column type
reaches its expected call at no higher SNR than on `gauss` plus one ladder step, and (ii) the
`noise` columns' rate of any non-"none" call is within the binomial 95% interval of the `gauss`
rate. A shape that fails is reported, and that source is not compared until the ruler is fixed
or the failure is shown not to affect structural calls.

## 5b. Random directions (basis control)

### `atlas/columns.py`

A column source maps the forward's per-layer tensors to the `[T, M_l]` matrix the Atlas sees.

```
class Source:              # interface
    name: str
    width(l) -> int        # columns at layer l (constant per source in practice)
    hook: "pre" | "post" | "resid"
    __call__(l, H) -> Tensor[T, width]   # H: the hooked tensor of layer l, [T, d]
```

- `neurons_pre`: identity at the pre-GELU hook (today's atlas; bitwise unchanged).
- `neurons_post`: identity at the post-GELU hook (output of `layer.intermediate`).
- `randdir(M, seed)`: post-GELU hook; `H @ R_l` with `R_l` `[d_ff, M]` Gaussian columns
  normalised to unit length, generated on CPU in fp32 from
  `torch.Generator().manual_seed(seed * 1000 + l)` and cast once to the model dtype on the
  device; the product accumulates in fp32 and is returned in fp32. `R_l` is saved alongside the
  atlas (`directions.pt`) and its sha256 goes into `run.json`.
- `shuffled` stays a model control (`--control shuffled`), combined with `neurons_post`.

### `scripts/run_atlas.py`

`--columns SPEC[,SPEC...]` with `SPEC` in `neurons_pre` (default), `neurons_post`,
`randdir:M:seed`, `sae:PATH` (5c). One forward feeds one Atlas per source. Each source gets its own
pilot edges and seeds (from the same pilot forward), its own output (`atlas.pt` when there is one
source, `atlas_<name>.pt` otherwise; `shard_<name>.pt` under `--shard`), and its own digests in
`run.json`. Overflow redo: after a batch, every Atlas reports its redo layers; one extra forward
runs with each Atlas active only on its own redo layers. Sharding, the pilot artifact (one entry
per source), `merge_atlas.py` (one call per source) and the gate apply per source unchanged.

Acceptance:

- `--columns neurons_pre` gives today's `atlas.pt`, bitwise.
- `--columns neurons_pre,neurons_post` gives, for `neurons_pre`, the same atlas as a run with
  that source alone (bitwise; shows that sharing the forward changes nothing), and a
  `neurons_post` atlas whose `top_val` equals `gelu` applied to the matching pre-GELU values in
  fp32 for every unit whose pre top-K is entirely above 0 (GELU is monotone there).
- `randdir:M:seed` reproduces bitwise across two runs and across `--shard 0/2, 1/2` + merge.
- With `M = d_ff` and `R_l = I` (test-only switch) the `randdir` atlas equals `neurons_post`.
- Throughput record: per-batch time of `neurons_pre,neurons_post,randdir:512:0` against
  `neurons_pre` alone on the same batches (`batches.jsonl`), to measure what a shared forward buys.

## 5c. InterPLM SAE latents

- Loader `atlas/sae.py`: reads an InterPLM SAE checkpoint and its metadata from a local path.
  The implementer must take the encoder formula, the input normalisation, the hooked tensor (which
  residual stream position, before or after the layer's final LayerNorm) and the layer indices
  from InterPLM's own code and model card, cite file and line in the docstring, and stop and
  report if any of these cannot be established from the source. No formula is to be inferred from
  the weight shapes alone.
- Hook `resid`: the output hidden state of `encoder.layer[l]`, as InterPLM defines it.
- Acceptance: on a stored batch, the loader's latents match latents computed by InterPLM's own
  package on the same input to fp32 rounding (one-off check, its log committed); the SAE source
  passes the same sharding gate as the other sources at 8M on CPU.
- Licence and download location of the SAE weights are recorded in `NOTICE` before use.

## Not in this unit

Sampling columns for the ruler, sealing the science predictions and decision rule, and running the
four-source comparison. Those follow once 5a's gate is recorded.
