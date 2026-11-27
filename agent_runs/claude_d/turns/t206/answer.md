The per-lag regression shows where each readout falls short. Values are rollout response per unit of dose change, original → E2, against the structural target:

| edit | lag 1 | lag 2 | lags 3–4 | lags 5–6 |
|---|---|---|---|---|
| insulin ×2 | -0.2 → -1.43 (target -2.0) | → -0.94 (target -2.4) | → about -0.9 (target -2.1, -1.7) | fades |
| insulin ×½ | → -1.49 | → -0.73 | weaker than ×2 | weaker than ×2 |
| NE ×½ | → 1.8 | → 1.8 | → 0.9, 1.9 | → 1.6, 3.4 |
| NE ×2 | → 0.54 | → -0.48 | ≈ 0 | ≈ 0 |

For insulin the immediate gain is the same per unit dose under doubling and halving, so the decoder is linear near the factual range. The halving readout is small mainly because the later lags fade. For NE, doubling pushes rates beyond the factual range, the state's dose representation saturates there, and a head edit cannot see the difference.

Even this partial response already exceeds the NLL budget. I need either more budget or a better signal-to-noise ratio. One option for budget: offline, the head scores better with z at the prior mean (NLL 1.5818) than with the four prior draws the tool uses (1.5955).

The prior is only used for head predictions, not for the state update. So shrinking the prior's standard deviation could recover up to 0.014 nats of factual fit, and it would also cut the z-driven part of the decoder noise. I am locating the z_prior tensors to check that this is feasible.

```python
print([ (n, SHAPES[n]) for n in NAMES if 'z_prior' in n or 'z_post' in n or n.startswith('core.z') ])
```