The arrays match what I need. XIN3 holds [ctx, e_k, u_k] at the 2,319 drug bundles of sel100. H3, OUT3 and Z3 hold the original model's state after each of the 5,218 bundles.

So I can rebuild the 7 prelude tokens at drug bundles: the time-decayed h plus slots, e_k, u_k and z_in(z). What I don't know is the attention head count and any scaling. The opening lines of ARCHITECTURE.md probably state them, so I'll print lines 0–30. I'll also check that the drug-bundle count matches the state rows.

```python
print('\n'.join(l[:200] for l in lines[0:30]))
dr = (A['has_insulin'] | A['has_norepinephrine'])
print('drug rows', dr.sum())
```