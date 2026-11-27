The empty-set replica matches the probe exactly, so the back half of the encoder is confirmed: GELU, mlp2, `size(log(1+n))` and the final LayerNorm. The vocab also gives the exact preprocessing rules:
- z = (x − median)/iqr, clipped to ±10;
- fine bin = 1 + the number of edges strictly below x;
- the padding rows of the categorical tables are zero.

What's left is the single-atom embedding for an insulin or norepinephrine record. It is a sum of the code, kind, ekind, categorical, bin and numeric terms, and only some of these may apply, with some indices unknown.

Next I'll pull the insulin and NE numeric entries (rate and dose medians, IQRs and edges). Then I'll search over which optional terms are included and which index each takes. The target is the probe's 20 exact u_k coordinates for insulin-only bundles at rates 0.5, 1, 2, 3, 4 and 6. Every rate matching to about 1e-3 would confirm the replica; no combination matching would mean a term is missing or the atom handling differs.

```python
num = {(d['code'], d['attribute']): d for d in V['numeric']}
for k in num:
    if k[0] in ('insulin','norepinephrine'): print(k, {a: num[k][a] for a in ('median','iqr')}, 'edges', num[k]['fine_bin_edges'])
# probe targets for insulin-only bundles: distinct rates
tgt = {}
for rv in np.unique(ri[m_io & np.isfinite(ri)]):
    mm = m_io & np.isclose(ri, rv); tgt[float(rv)] = U[mm][0]
print('insulin target rates:', sorted(tgt))
for t in ['embed.kq.weight','embed.text.weight','embed.kind.weight','embed.ekind.weight','embed.categorical.unit.weight','embed.categorical.rate_unit.weight']:
    w_ = load(t); print(t, 'row norms', w_.norm(dim=1).numpy().round(2))
```