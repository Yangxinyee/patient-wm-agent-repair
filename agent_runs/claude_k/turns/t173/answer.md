The vocab gives the exact conventions:
- z = (x − median)/IQR, clipped to [−10, 10]. Insulin rates above 11.5 units/h are therefore indistinguishable to the model, which will cap some x2 and x3 effects whatever I do.
- Fine bin = 1 + number of edges strictly below x. The edge lists contain repeats (insulin rate: 0.5×3, 1.0×4, 1.5×2, 2.0, 2.5, …).
- Insulin dose uses the same edges as rate. Norepinephrine dose ≈ 4.8 × rate.

A simple left/right bin-side swap does not reproduce the observed jump between rates 1.5 and 2.0, so the bug may be in the atom attributes themselves. Candidates are the recorded dose values or END-atom contents.

I'll look at the raw records for one insulin stay: the README's column description and the drug rows. Then I'll compare each atom `drug_atoms` builds with atoms I build by hand via `atom_vec` from those raw values. If the recorded dose is not equal to the rate, or the END atoms carry attributes, that would explain the discrepancy.

```python
rd=open('/data/README.md').read(); print(rd[:2500])
import pandas as pd
sid_j0=int(sid8[j0]); t_j0=int(np.floor(hr8[j0]))
it=pd.read_csv('/data/train.csv', nrows=400000)
rows_=it[it[it.columns[0]]==sid_j0] if it.columns[0]!='stay_id' else it[it.stay_id==sid_j0]
print(it.columns.tolist())
print(rows_.head(3).to_string()[:800])
dr=rows_[rows_.astype(str).apply(lambda r: ('insulin' in ' '.join(r.values)) or ('norepi' in ' '.join(r.values)), axis=1)]
print('t_j0',t_j0); print(dr.head(12).to_string()[:2500])
```