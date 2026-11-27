`encode_bundles` takes the atom table plus an integer row index; I passed the bundle DataFrame, which caused the dtype error. The atom table has stay, hour, event, value, rate, dose, kind, minute and row (bundle index), so the dose edits only need rate and dose scaled on insulin atoms.

This run:
- retries with `at['row']` as int64 and checks that it reproduces the stored e and u;
- builds the ×2 and ×0.5 insulin versions of dsE;
- counts bundle types (insulin, NE, measurement).

If the re-encoding is exact, the next run collects `n3_0` under the three dose versions to fit the gate and the rate decoder.

```python
rows_=at['row'].values.astype(np.int64)
e2,u2=encode_bundles(at, rows_)
e2=np.asarray(e2); u2=np.asarray(u2)
print('shapes', e2.shape, u2.shape, '| err e', float(np.abs(e2-np.asarray(dsE['e'])).max()), 'u', float(np.abs(u2-np.asarray(dsE['u'])).max()))
nb=len(bd)
insr=np.zeros(nb); ner=np.zeros(nb); nmeas=np.zeros(nb)
a_=at[at.event=='insulin']; insr[a_['row'].values]=a_['rate'].values
a_=at[at.event=='norepinephrine']; ner[a_['row'].values]=a_['rate'].values
a_=at[at.kind==0]; np.add.at(nmeas, a_['row'].values, 1)
print('bundles', nb, 'ins', int((insr>0).sum()), 'ne', int((ner>0).sum()), 'both', int(((insr>0)&(ner>0)).sum()), 'meas', int((nmeas>0).sum()), 'other', int(((insr==0)&(ner==0)&(nmeas==0)).sum()))
print('kinds', at.groupby('event')['kind'].first().to_dict())
```