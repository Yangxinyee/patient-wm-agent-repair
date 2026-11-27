Of the non-drug bundles, 2,473 are pure-measurement and encoded. 426 failed. These are probably END-only bundles at times when a drug stopped (which have no row in the event table) and admission bundles. I'll break the failures down by type before deciding how to encode them.

```python
fr = np.where(src == 'fail')[0]
kinds_f = {}
for i in fr:
    g = grp_all.get((st[i], hr[i]))
    key = 'none' if g is None else '+'.join(sorted(set(g.event)))
    kinds_f[key] = kinds_f.get(key, 0) + 1
print(sorted(kinds_f.items(), key=lambda t: -t[1])[:10])
print('frac hour non-integer among fails', np.mean(np.abs(hr[fr]-np.round(hr[fr]))>1e-6))
```