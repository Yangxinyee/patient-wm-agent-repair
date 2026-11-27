gen_rows_all gives rows for all four reading codes, and `allrows` already holds 158,909 factual rows. That is the factual covariance I need for the minimum-variance decoder. For the dose-response side I must recompute head inputs with counterfactual doses: double or halve every infusion from hour 12, keep the recorded readings, and take the change in with_code per row.

The cleanest way is to edit the stay's dataframe and rebuild bundles, so first I need the dataframe format and how gen_rows_all takes precomputed states (St). Check: rows rebuilt from my own run_stay states with unchanged doses should equal gen_rows_all's default output exactly.

```python
from collections import Counter
print(Counter([r[0] for r in allrows]).most_common(4))
ust = set(r[1] for r in allrows); print('allrows stays', len(ust), 'overlap G3', len(ust & set(G3.keys())))
df0 = G3[s0]; print(list(df0.columns)); 
print(df0[df0.event.isin(['insulin','norepinephrine'])].head(3).to_string()[:600] if 'event' in df0 else df0.head(3))
print(df0.event.value_counts().to_dict())
B0_ = bundles_of_stay(df0, RU0); St0_ = run_stay(B0_); print(type(St0_).__name__, len(St0_) if hasattr(St0_,'__len__') else '', type(St0_[0]).__name__ if hasattr(St0_,'__getitem__') else '')
ra2 = gen_rows_all(df0, St=St0_); print('max diff wc', max(np.abs(a[8]-b[8]).max() for a,b in zip(ra, ra2)))
```