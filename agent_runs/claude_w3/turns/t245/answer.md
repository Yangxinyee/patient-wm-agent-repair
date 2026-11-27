Every dimension of h varies across bundles. The smallest spread is about 2.4 (dimensions 49, 149, 9, 61 and 185), against a median of 8.7. The per-step retention `a_bar` is about 0.35 for all dimensions, so any memory currently carried by h runs through E and `b_in`, not through the gate.

Next I'll list the normalisation and head tensor names. Then I can measure how strongly each low-spread dimension is read downstream, and test whether one or two of them can be freed for a memory channel without harming the factual fit.

```python
nm=[n for n in NAMES if ('norm' in n and n.endswith('weight')) or 'z_prior' in n or 'z_post' in n or 'value_bin' in n or 'z_in' in n or 'z_head' in n]
print(len(nm)); print(' '.join(nm)[:1800])
```