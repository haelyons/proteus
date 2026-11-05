# Measurement #3: what a unit writes, and what silencing it does (written 2026-11-05, before it runs)

`scripts/unit_writes.py`, ESM-2 8M (fp32, CPU), 400 random non-held-out Swiss-Prot proteins (seed 0, cropped
to 1,022), 50 random units per layer (6 x 50 = 300). For each unit, three tokens: its largest pre-GELU value
over the sample (positive extreme), the token nearest its median, its smallest value (negative extreme).
Per case:
- h, GELU(h);
- written norm |GELU(h)| x ||W_out[:, u]|| against the residual it is added to (the attention output at
  that position, which `EsmOutput` adds the FFN write to);
- silencing: that unit's GELU output set to 0 at that one position, effect on the model's logits at that
  position: delta log p(true residue) and KL(p || p_silenced);
- travel: the largest KL at any other position of the protein;
- baseline: a random other unit of the same layer silenced at the same position, a stated baseline rather than a
  matched one (no other unit is near a positive extreme there).

Generalises the single trace in `assessment_20261104.md` section 5 (layer 2 unit 826: written 87.6 vs
residual 82.4 at h = 18; log p(Asp) -2.80 when silenced; nothing written at h = -20.5).

| quantity (median over the 50 units of a layer, every layer unless stated) | positive extreme | median token | negative extreme |
|---|---|---|---|
| GELU(h) | = h, 2-8 | -0.17 to -0.05 (h between -2 and -0.3) | \|GELU\| < 0.01; layer 5 < 0.001 |
| written / residual | 0.03-0.3 | 0.001-0.01 | < 0.001 |
| KL at the position, silenced | 1e-4 to 1e-2 | 1e-7 to 1e-5 | < 1e-7 |
| \|delta log p(true)\| | 1e-3 to 1e-1 | < 1e-3 | < 1e-4 |
| KL, unit / baseline unit at the same position | 10-1000x | ~1x | < 1x |

- Tail: >= 10% of units have |delta log p(true)| > 0.1 at their positive extreme (a strong unit like 826 is
  not unique), and < 2% at the negative extreme.
- Travel: exactly 0 for layer 5 (nothing after the last FFN mixes positions: final LayerNorm and LM head
  are per position); for layers 0-4 at the positive extreme, median travel KL 1e-6 to 1e-4, below the
  KL at the position itself.
- What would contradict "the negative pole is the off state": a layer where the median |GELU| at the
  negative extreme exceeds 0.05, or where silencing there has a median KL above the median token's.
