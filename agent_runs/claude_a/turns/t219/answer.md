The bytecode shows `atom_vectors` does read `rate` and `dose`, through the z-score and bin helpers (`z32`, `fb32`) and the unit maps. So caching by index is not the explanation for the identical grid encodings. Two other suspects:
- my grid construction;
- the action/measurement split, which may rely on a column that is not set the way I assume.

Next I will test directly:
- whether the stored u is non-zero and differs across factual insulin-only bundles;
- the factual R² of a linear rate decoder on all 192 dims of u;
- whether `atom_vectors` gives different vectors and action flags for my grid copies.

How I will read it:
- A high R² on full u means the rate is encoded, so head 3's 48-dim projection is the bottleneck.
- Identical atom vectors across the grid mean the grid input is malformed, for example a unit field missing.

```python
uE=dsE['u'].numpy(); m=(RS['fac']>0)&(~isNE)
print('||u|| ins-only', np.round(np.percentile(np.linalg.norm(uE[m],axis=1),[5,50,95]),2), '| meas-only', np.round(np.percentile(np.linalg.norm(uE[(RS['fac']==0)&(~isNE)],axis=1),[5,50,95]),2))
X1=np.c_[uE[m],np.ones(m.sum())]; w_=np.linalg.lstsq(X1,RS['fac'][m],rcond=None)[0]
print('rate from full u: R2', round(1-np.var(X1@w_-RS['fac'][m])/np.var(RS['fac'][m]),4), 'resid sd', round(np.std(X1@w_-RS['fac'][m]),3))
av,isa=atom_vectors(sg, {})
av=np.asarray(av); print('grid atoms', len(sg), 'is_action', np.asarray(isa).astype(int).tolist(), '| av diff vs first', np.round(np.linalg.norm(av-av[0],axis=1),3).tolist())
print(sg[['event','rate','dose','kind','row']].head(3).to_string())
```