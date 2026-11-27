Lagged dose information fades fast in the state:

| Target | CV R² |
|---|---|
| Recorded rate, lag 1 / 2 / 3 | 0.79 / 0.75 / 0.68 |
| Recorded rate, lags 4–8 | about 0.62 |
| Deviation from policy, lag 1 / 2 / 3 / 4 | 0.46 / 0.27 / 0.14 / 0.05 |

The lagged rates stay decodable only because they can be inferred from the glucose history (the policy). The state keeps real information on dose deviations for only about 1–2 bundles-hours.

Combined with the original rollout's linear growth, this means the model carries a drawn glucose shift almost fully forward and remembers actual doses for only about 2 hours.

**What a head-only correction can and cannot do.** It cannot produce the true concave profile. That would need a positive "wearing off" term at lags 3–10, and the state cannot supply it. It could produce a linear-growth profile, calibrated to the 12-hour mean. That would err by about 15–30% on the start-6 and start-16 windows.

To do better, or even to test reliably how decoders respond to doubled doses, I need counterfactual states. That requires a local re-implementation.

**Plan.** Rebuild the core step from scratch and verify it against the fwd_3 states:
- the atom embedding, encoders and z posterior;
- then the prelude, blocks and coda.

The main unknown on the input side is which categorical rows each atom uses. Row norms of the categorical, kind, ekind, kq and text tables should show which rows were ever trained, which narrows the choices. If some rows are near zero or at initialisation scale, they are unused and the assignment becomes nearly determined.

```python
for nm_ in ['embed.categorical.action.weight','embed.categorical.discharged_to.weight','embed.categorical.rate_unit.weight','embed.categorical.route.weight','embed.categorical.status.weight','embed.categorical.unit.weight','embed.kq.weight','embed.text.weight','embed.kind.weight','embed.ekind.weight','embed.code.weight','embed.bins.value.weight','embed.bins.rate.weight','embed.bins.dose.weight','last_status.weight','type_emb.weight']:
    W_ = load(nm_).numpy()
    print(f'{nm_:40s}', np.round(np.linalg.norm(W_, axis=1), 2))
nw_ = load('embed.numeric.weight').numpy(); print('numeric col norms', np.round(np.linalg.norm(nw_, axis=0), 2), 'bias norm', round(float(np.linalg.norm(load('embed.numeric.bias').numpy())), 2))
print('records rate_unit/dose_unit values:', trs['rate_unit'].dropna().unique()[:6], trs['dose_unit'].dropna().unique()[:6])
```