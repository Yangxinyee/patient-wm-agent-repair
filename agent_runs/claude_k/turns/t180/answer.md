None of the END variants fits: the best error is still 0.42, far worse than the 0.03 for single atoms. So either END atoms are not where I assume, or they look quite different.

The states file records `n_atoms` for each bundle, so I can check directly. Bundles with an insulin start at hour t and an insulin record at t−1 (no norepinephrine) should show 2 atoms if an END atom shares the bundle, and 1 if not.

I'll print the n_atoms distribution for each composition. I'll also test the 'no END atom' variant, plus the END variants with categorical rows, which the last search ranked low. If n_atoms = 1 and the start-only encoding fits, END atoms simply don't appear in these bundles.

```python
na8=d8o['n_atoms']
grpdf=pd.DataFrame(dict(n=na8[ok_], ins_now=(cur_i>0).astype(int), ne_now=(cur_n>0).astype(int), ins_prev=(prev_i>0).astype(int), ne_prev=(prev_n>0).astype(int)))
print(grpdf.groupby(['ins_now','ne_now','ins_prev','ne_prev']).n.agg(['count','min','max','mean']).round(2))
# no-END variant on start+prev bundles
pI=np.array([p_ins_@np.asarray(setenc_my([datom(8,a)],'encode_action')) for a in rc]); pN=np.array([p_ne_@np.asarray(setenc_my([datom(8,a)],'encode_action')) for a in rc])
print('no END atom: rmseI',np.round(np.sqrt(np.mean((pI-yI3)**2)),3),'rmseN',np.round(np.sqrt(np.mean((pN-yN3)**2)),3))
print([ (np.round(x[:3],3),x[3]) for x in resE if x[3][1]==True][:4])
# non-drug bundles atoms count sanity
print('n_atoms for drug bundles overall', np.bincount(na8[hi|hn].astype(int)))
```