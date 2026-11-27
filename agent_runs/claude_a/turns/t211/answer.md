`encode_bundles` returned one row per entry of the index I gave it (25,774, the atom count). So its second argument is the list of bundle ids to encode, not a per-atom index. I will pass `bd['row']` (13,036 bundle ids) instead, and run the same checks:
- exact reproduction of the stored e and u;
- counts of insulin, NE, both-drug, measurement and other bundles;
- each event's kind code.

If the reproduction is exact, synthetic dose versions only need the insulin atoms' rate and dose scaled before encoding.

```python
rows_=bd['row'].values.astype(np.int64)
e2,u2=encode_bundles(at, rows_)
e2=np.asarray(e2); u2=np.asarray(u2)
print('shapes', e2.shape, u2.shape, '| err e', float(np.abs(e2-np.asarray(dsE['e'])).max()), 'u', float(np.abs(u2-np.asarray(dsE['u'])).max()))
nb=len(bd); ridx={r:i for i,r in enumerate(bd['row'].values)}
insr=np.zeros(nb); ner=np.zeros(nb); nmeas=np.zeros(nb)
a_=at[at.event=='insulin']; insr[[ridx[r] for r in a_['row'].values]]=a_['rate'].values
a_=at[at.event=='norepinephrine']; ner[[ridx[r] for r in a_['row'].values]]=a_['rate'].values
a_=at[at.kind==0]; np.add.at(nmeas, [ridx[r] for r in a_['row'].values], 1)
print('bundles', nb, 'ins', int((insr>0).sum()), 'ne', int((ner>0).sum()), 'both', int(((insr>0)&(ner>0)).sum()), 'meas', int((nmeas>0).sum()), 'other', int(((insr==0)&(ner==0)&(nmeas==0)).sum()))
print('kinds', at.groupby('event')['kind'].first().to_dict())
```