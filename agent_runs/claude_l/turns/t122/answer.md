My replica treats each infusion END atom as a copy of the record carrying its rate and dose. The run-105 check showed END atoms go to the observation encoder (e), not the action encoder. So the memory M does hold past doses, lagged by one hour, in the e vectors of the drug bundles.

Two mechanisms follow:
- **Lag 1:** the dose effect can only come through the current bundle's u_k.
- **Lags ≥ 2:** the effect could come from a cross-attention head reading END-atom doses in M.

Either design needs an exact replica of the e encoder. Mine is still off by about 1 on every bundle that contains measurements, while the END-only bundle is nearly right (0.13). So the bug is probably in how measurement atoms are featurized: the value z-score or bins, the unit or ekind index, or the `kq` table.

I'll print the vocab structure and the `embed`/`encode` tensor shapes to look for that detail.

```python
print(list(VOCAB.keys()))
for k,v in VOCAB.items():
    if k in ('codes',): continue
    s=json.dumps(v); print('##',k, len(s), s[:300])
print('; '.join('%s%s'%(n,tuple(SHAPES[n])) for n in NAMES if n.startswith('embed') or n.startswith('encode')))
```