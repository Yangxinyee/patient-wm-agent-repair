The existing decoders from session 6 are not usable as they stand:
- **`vI`:** responds to the insulin edit with only +1.5 against a target of −11, and its within-cell factual noise sd is 14–21.
- **`vN`:** its response has the right sign but is ten times too small, and its noise is large.

The ridge decoders from session 5, with errors of about 0.57 u/h for insulin and 0.19 for NE, gave a much better trade-off. Next I'll find which kernel arrays line up with `Xf`, `EDR` and the session-5 decoder dictionary, so I can rebuild a decoder from aligned data. I'm printing names and shapes only.

```python
cands = {}
for k_, v_ in g.items():
    if k_.startswith('_') or not hasattr(v_, 'shape') or v_.ndim == 0: continue
    if v_.shape[0] in (70738, 20313, 13051, 21168, 17722, 39178): cands[k_] = tuple(v_.shape)
print(cands)
print('DEC' in g, [k_ for k_ in g if k_.upper().startswith('DEC')][:10])
print('w_ins' in g, 'w_ne' in g, [k_ for k_ in g if k_.startswith(('w_','dec_','ridge'))][:10])
```