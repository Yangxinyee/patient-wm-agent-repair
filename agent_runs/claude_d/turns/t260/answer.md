The tensors needed for the z-posterior check all exist. These are core.flow.a, core.out_norm, gap.0/gap.2, the z_post MLP (input [ctx, e_k, u_k], output 64 = 32 means + 32 log-sds), and the encode_action set encoder.

One unknown remains before I can rebuild drug atom vectors exactly: which rows of the categorical tables (unit, rate_unit, route, status, action) and of `embed.kq` a drug atom uses. VOCAB does not list their ids.

I will print the sizes of those tables. If they are small, I can either find the right ids by matching the z posterior, or orthogonalize the gate against all their rows so the gate is exact whatever the ids are.

```python
print({n: SHAPES[n] for n in NAMES if n.startswith('embed.categorical') or n in ('embed.kq.weight','embed.text.weight','embed.kind.weight','embed.ekind.weight','embed.code.weight')})
print('n atoms per minute check:', TRAIN[TRAIN.event.isin(['insulin','norepinephrine'])].groupby(['stay_id','hour']).size().value_counts().head(4).to_dict())
print(TRAIN[TRAIN.event=='norepinephrine'][['rate_unit','dose_unit']].drop_duplicates().values.tolist(), TRAIN[TRAIN.event=='insulin'][['rate_unit','dose_unit']].drop_duplicates().values.tolist())
```