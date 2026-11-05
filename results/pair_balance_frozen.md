# Pair balance (layer 1, units 86 and 1200): frozen test (2026-11-05, before the confirmatory run)

## Already inspected (everything below was seen before freezing)

- Weights: input cosine 0.81, output cosine -0.661, |w_out| 1.68 / 1.12.
- First experiment (400 held-out proteins, seed 0; `results/valine_pair_20261105/`): opposed effects at the
  position (cosine -0.78), sub-additive joint silencing (0.40 vs controls 0.80), silencing 86 lowers E/Q/G
  and raises I.
- Discovery (`scripts/pair_balance.py discover`; 600 non-held-out proteins, seed 2; 215,406 tokens):
  - fit on tokens where either unit is above its q99 (2,756): h1200 = 1.112 + 0.656 h86, R^2 0.31; residual
    q10 / q90 = -0.467 / 0.404;
  - top tokens: 85% strand, 81% buried tercile;
  - 86-heavy departures: centre D (log2 +3.0), S at centre, H at +1, L/V/Y/F at -3, D at +2; more loop
    (19% vs 10%) and less buried (75% vs 81%);
  - 1200-heavy departures: K/S/Q at -2, G at -1, Q at -3;
  - graded interventions at 30 positions, read at the masked +-3 neighbours: joint within 25% of the vector
    sum of the single effects (cosine of the single effects -0.73); effects on log p(true) ~1e-3 nats.
- Smoke runs of `pair_balance.py` itself: only the discovery run above.

## Proposal

The pair detects "buried beta-strand position". Its balance encodes the identity and polarity of the centre
residue and its N-terminal context. Because the writes are opposed, the balance shifts predictions at nearby
masked positions along a polar/hydrophobic axis, and this natural variation helps masked prediction there.

## Independent reviewer's alternative (agent given only the proposal and the record above)

The residual h1200 - 1.11 - 0.656 h86 is, before GELU, a fixed linear direction applied to the layer-1 input,
which is dominated by the centre token's embedding plus local attention terms. So the D/S and K/Q/G
signatures are properties of the weights, not a variable the model computes. The shared component reads a
local, valine-rich, beta-branched sequence pattern that correlates with buried strand without being
structural. Downstream, the residual adds a fixed push along the polar axis, which "helps" only where the true
neighbours happen to be polar; D/S-centred windows have more polar neighbours. The low R^2 is partly range
restriction (tokens selected because one unit or the other is extreme).

The reviewer noted that my planned part (a), recurrence, would pass under the alternative, and that part (b)
was confounded by neighbour composition, with a control whose direction was not matched.

## Frozen test (fresh proteins: held-out with descriptors, excluding the first experiment's 400; seed 1; 600)

1. **Natural vs injected departures at matched neighbour composition (primary; the reviewer's observation).**
   - Natural: at 86-heavy departure sites (unmasked residual <= -0.467), in the forward with the +-3
     neighbours masked, set 1200's post-GELU value to the fitted one, GELU(1.112 + 0.656 h86). The gain is
     log p(true) at the masked neighbours with the natural value minus with the fitted one.
   - Injected: at top sites with |residual| <= 0.1, lower 1200 by a residual borrowed (seeded draw) from the
     natural departures, i.e. post-GELU set to GELU(h1200 + r). The gain is with injection minus without.
   - Regression of the gain (mean over the six neighbours) on: natural (0/1), polar fraction of the true +-3
     neighbours (D E K R N Q S T H), and centre D/S (0/1); bootstrap 95% interval over sites.
   - Proposal predicts: natural coefficient > 0 (interval above 0), and the D/S advantage surviving polarity.
   - Alternative predicts: natural coefficient ~ 0; gain explained by polar fraction at the same slope for
     natural and injected sites; D/S advantage gone once polarity is conditioned on.
   - My prediction: the alternative's (natural coefficient interval includes 0). The discovery effects are
     ~1e-3 nats, and the weights make a fixed readout plausible.
2. **Recurrence (reported, given little weight):** 86-heavy departures with centre D/S at >= 3x the base share.
   Expected to pass under either explanation.
3. **Context transplant ("buried strand" vs local sequence).**
   - Take 40 top sites (either unit above its discovery q99.9) and insert their +-3 window (7 residues) into
     other fresh proteins at host positions whose AlphaFold state is strand, helix or loop (5 hosts each).
   - Measure the retained activation: pre-GELU at the inserted centre divided by the original.
   - Proposal ("reads buried strand"): retention clearly higher in strand hosts than in helix / loop hosts.
   - Alternative (local sequence): similar retention across host states.
   - My prediction: similar retention (median ratio within 0.2 across host states), i.e. local sequence.
4. **Perturbation-matched control,** reported alongside 1: another layer-1 unit at the same site, changed by the
   same write norm in the same masked forward (direction not matched). The injection in 1 is the
   direction-matched control.

Whatever happens, the result is reported. If the alternative holds, the account will say: the pair's balance is
a fixed readout of local sequence, and its downstream "help" is explained by neighbour composition. That
explains what the balance *is*, but not a computation beyond the input.

### Amendment after the smoke (discovery proteins only; 40 proteins, 8 sites per kind, 3 transplant sites), before the fresh run

Seen in the smoke:
- the code runs;
- natural / injected |gain| medians 0.0014 / 0.0047 against matched-control |change| 0.0024 / 0.0022;
- the transplant measure (pre-GELU ratio to the original) is unstable: ratios of -2.2 to -0.03, because a
  site chosen for one unit can have a near-zero or negative original value for the other.

Changed: each unit is evaluated only at sites where it was above its discovery q99.9, and retention is the
fraction of transplants that stay above its discovery q99 (plus the median pre-GELU value), per host state.
The observation and both explanations' predictions are unchanged: the proposal predicts higher retention in
strand hosts; the alternative predicts similar retention across host states.
