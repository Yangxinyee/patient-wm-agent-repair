The layout is confirmed: out equals core.out_norm(mean over the 4 slots of h) to about 1e-6. The states file is slot-major and the LayerNorm epsilon is 1e-5.

Next I check the z posterior mean of each bundle against my own encoding. It depends only on:
- the previous state (known exactly from the states file), after the time flow;
- the time gap;
- my own encodings of the bundle's atoms and actions.

The embedding has several small lookup tables whose index conventions the documentation leaves open: unit, rate_unit, route, action, status, discharged_to, kq and text. Rows that are exactly zero, as for a padding index, or that look untrained would show which indices are used.

So this run does three things:
- print the row norms of every small embedding table;
- inspect a drug bundle in my parsed data, to see whether END atoms are included;
- check that the states rows line up one-to-one with my bundles.

If the row counts per stay match the bundle counts, I can index the states directly.

```python
for n_ in ['embed.kind.weight','embed.ekind.weight','embed.categorical.action.weight','embed.categorical.discharged_to.weight',
           'embed.categorical.rate_unit.weight','embed.categorical.route.weight','embed.categorical.status.weight',
           'embed.categorical.unit.weight','embed.kq.weight','embed.text.weight','embed.code.weight','type_emb.weight','last_status.weight']:
    print(n_, np.round(np.linalg.norm(W[n_], axis=1), 3))
for n_ in ['embed.bins.value.weight','embed.bins.dose.weight','embed.bins.rate.weight']:
    print(n_, np.round(np.linalg.norm(W[n_], axis=1), 2))
print('numeric col norms', np.round(np.linalg.norm(W['embed.numeric.weight'], axis=0), 3))
print('last_numeric col norms', np.round(np.linalg.norm(W['last_numeric.weight'], axis=0), 3))
sid0 = list(bundles_B.keys())[0]
for b_ in bundles_B[sid0][:12]:
    if any(a.get('code') in ('insulin','norepinephrine') for a in b_[1]): print(b_)
cnt_rows = pd.Series(Z['stay_id']).value_counts()
mm = [(s_, cnt_rows.get(s_, 0), len(bundles_B[s_])) for s_ in list(bundles_B.keys())[:100]]
print('rows==bundles for all:', all(a==b for _, a, b in mm), mm[:5])
print('hours first stay', Z['hour'][Z['stay_id']==sid0][:12], Z['n_atoms'][Z['stay_id']==sid0][:12])
```