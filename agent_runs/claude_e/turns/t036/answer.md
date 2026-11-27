Norepinephrine shows the same gap as insulin, but worse. On the tuning rows my estimate of the truth is +4.5 mmHg for x2 and −2.3 mmHg for x1/2. The estimated effect builds to about +5 mmHg within 4–5 hours and then plateaus. The model shows −0.15 and −0.40, flat over time. So it has essentially no norepinephrine effect, while its insulin effect is about 10% of the truth and too slow.

My current estimates of the four true readouts on these 100 tuning stays:

| edit | estimated truth | original model |
|---|---|---|
| insulin x2 | about −13.5 mg/dL | −1.6 |
| insulin x1/2 | about +6.8 mg/dL | +0.7 |
| norepinephrine x2 | about +4.5 mmHg | −0.15 |
| norepinephrine x1/2 | about −2.3 mmHg | −0.40 |

To engineer an edit that adds the right dose → glucose/MAP pathway, I need to reason about how changes propagate through the network. That means a local re-implementation, checked against the tool's states.

I'll save these conclusions to the notebook. Then I'll inspect the small "structural" tensors to pin down conventions before writing the forward pass:
- the time-flow decay rates (`core.flow.a`);
- `delta_in` and `a_in`;
- `core.slots`;
- the norms of the categorical embedding rows, since unused rows likely stay near initialisation.

```python
with open('/work/notes.md','a') as f: f.write("""
- NE effect on map_arterial (LP, FE band x prev rate, deconvolved): direct per 1 mcg/kg/min, lags1-6: [3.36 2.55 1.84 0.94 0.50 0.20], ~0 after; sum ~9.4. lag-1 effect same at low/high rates (linear).
  Titration uses arterial MAP band if present (else cuff): <65 up ~+0.12 log, 65-75 hold, >75 down ~-0.2; noise sd ~0.3 log. Arterial line in ~48% stays (95% hours, random).
- Truth est on sel1 rows: ins x2 -13.3..-14.3, half +6.7..+7.1; NE x2 +4.53, half -2.27. Model: -1.60, +0.70, -0.15, -0.40.
- Model's insulin shift ramps slowly (-0.4 at 1h ... -3 at 12h after first dose) vs truth (-3, -6.8, -9.7, -11.9 ... -20 at 12h). NE model ~0 flat.
- fwd 2: states of original on sel1 in /work/fwd_2 (5116 bundles). Bundles = distinct times; END atoms at record end (s+1.0833); final bundle 25.1667 (admission ENDs). has_insulin = new insulin record.
""")
def show(n):
    t = load_original(n); print(n, tuple(t.shape), 'mean %.4f std %.4f min %.4f max %.4f'%(t.mean(), t.std(), t.min(), t.max()))
for n in ['core.flow.a','core.a_in','core.delta_in','core.slots','embed.numeric.bias']: show(n)
fa = load_original('core.flow.a').numpy(); tau = 1/np.exp(fa); print('flow timescales (s): min %.0f max %.0f'%(tau.min(), tau.max()), np.quantile(tau,[.1,.5,.9]).round(0))
for n in ['embed.code.weight','embed.kind.weight','embed.ekind.weight','embed.kq.weight','embed.text.weight','embed.categorical.status.weight','embed.categorical.unit.weight','embed.categorical.route.weight','embed.categorical.action.weight','embed.categorical.rate_unit.weight','embed.categorical.discharged_to.weight','embed.bins.value.weight','embed.bins.dose.weight','embed.bins.rate.weight','type_emb.weight','last_status.weight']:
    t = load_original(n).numpy(); print(n, 'row norms', np.linalg.norm(t,axis=1).round(2))
print('embed.numeric.weight col norms', np.linalg.norm(load_original('embed.numeric.weight').numpy(),axis=0).round(2))
print('last_numeric.weight col norms', np.linalg.norm(load_original('last_numeric.weight').numpy(),axis=0).round(2))
```