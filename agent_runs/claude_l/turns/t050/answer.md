`vocab.json` gives the input conventions:
- 13 codes. The 4 measurement codes have ids 5, 6, 7 and 10; insulin is 8 and norepinephrine 9.
- Three atom kinds: WORLD, REVEAL (unused) and END.
- Each value is z-scored per code as (x − median)/IQR, clipped to [−10, 10].
- Fine bins are 1 + the number of edges below the value.
- The predicted mean is a bin-centre average. For glucose_finger it is bounded between 73 and 272.

Before rebuilding the reimplementation, I want to see the structures I built last time: the per-stay bundle dictionary `BUN` and the raw event frame `tr`. Reusing their atom layout saves runs, and I should check it against the vocab before trusting it. I also want the full numeric table for the drug rate and dose rows (median, IQR and edges).

```python
num = V['numeric']
for e_ in num: print(e_['code'], e_['attribute'], e_['median'], e_['iqr'], e_['fine_bin_edges'])
print(type(tr), getattr(tr, 'shape', None)); print(tr.head(8).to_string())
b0 = BUN[batch1[0]] if batch1[0] in BUN else BUN[list(BUN)[0]]
print(type(b0), (len(b0) if hasattr(b0, '__len__') else ''))
try:
    print(b0[:4] if isinstance(b0, list) else {k_: (type(v_), getattr(v_, 'shape', None)) for k_, v_ in (b0.items() if isinstance(b0, dict) else [])})
except Exception as ex: print('err', ex)
S2h = S2['hour'][S2['stay_id'] == batch1[0]]; print('bundle hours stay0', np.round(S2h[:12], 4), S2['n_atoms'][S2['stay_id'] == batch1[0]][:12])
```