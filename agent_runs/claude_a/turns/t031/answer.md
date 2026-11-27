Run 27 shows a clear asymmetry between the drugs:
- The insulin dose is well represented in the state: R² 0.80 for the current rate, and lagged rates 0.74/0.66/0.60, though these partly overlap with the current rate and glucose.
- The norepinephrine dose is poorly represented: R² 0.35 for the rate and only 0.05 for its exogenous log-ratio change.

So a head-level correction could work for insulin, but norepinephrine needs its dose carried into the state, an input/core-level change.

To design and check any edit offline, I'll start reimplementing the model on CPU, beginning with the heads. Those can be validated against the factual predicted means using the states I already have. First I'll inspect the small tensors that settle ambiguous details: the `last_numeric` columns (8 inputs, not the 4 documented), `type_emb`, `code_bias` and the flow timescales.

```python
W = {n: load_original(n).numpy() if hasattr(load_original(n),'numpy') else np.asarray(load_original(n)) for n in NAMES}
print('last_numeric col norms', np.linalg.norm(W['last_numeric.weight'],axis=0).round(3))
print('numeric col norms', np.linalg.norm(W['embed.numeric.weight'],axis=0).round(3))
print('type_emb row norms', np.linalg.norm(W['type_emb.weight'],axis=1).round(2))
print('kq', np.linalg.norm(W['embed.kq.weight'],axis=1).round(2), 'text', np.linalg.norm(W['embed.text.weight'],axis=1).round(2))
for n in ['embed.categorical.unit.weight','embed.categorical.rate_unit.weight','embed.categorical.action.weight','embed.categorical.route.weight','embed.categorical.status.weight','embed.categorical.discharged_to.weight','embed.kind.weight','embed.ekind.weight','last_status.weight','embed.bins.value.weight','embed.bins.rate.weight','embed.bins.dose.weight']:
    print(n, np.linalg.norm(W[n],axis=1).round(2))
print('code_bias', W['code_bias'].round(2))
fa = W['core.flow.a']; print('flow timescales (s) min/max', (1/np.exp(fa)).min(), (1/np.exp(fa)).max())
print('delta_in', W['core.delta_in'][:5], 'a_in', W['core.a_in'][:5])
```