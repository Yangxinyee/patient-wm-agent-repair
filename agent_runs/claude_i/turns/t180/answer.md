My rebuilt encoder does not yet reproduce the tool's z exactly: median error 0.04, with 10% of bundles off by more than 0.5. Earlier sessions reached an exact match, so something is wrong in one of three places: the atom recipe, the bundle-to-row mapping, or the precomputed ctx.

I'll split the error by bundle type: measurement-only, drug starts, end-only and visit events. That shows which part of the recipe is wrong. If drug-start bundles are exact and only one other type errs, the fix is local. If all types err, ctx or the row mapping is the problem.

```python
typ_b = np.array(['']*5156, dtype=object)
for key,atoms in atoms_by_bundle.items():
    i = rowkey.get((int(key[0]), int(key[1])))
    cs = set((a['code'],a.get('kind',0)) for a in atoms)
    t_ = []
    if any(c in ('insulin','norepinephrine') and k==0 for c,k in cs): t_.append('S')
    if any(k==2 for c,k in cs): t_.append('E')
    if any(c in ('glucose_finger','glucose_lab','map_arterial','map_cuff') for c,k in cs): t_.append('M')
    if any(c in ('hospital_admission','icu_admission') for c,k in cs): t_.append('V')
    typ_b[i] = ''.join(t_)
mx = np.abs(d_).max(1)
for t_ in sorted(set(typ_b)):
    w_ = typ_b==t_
    print(t_, w_.sum(), 'median', np.median(mx[w_]).round(4), '90%', np.quantile(mx[w_],0.9).round(4))
print('first rows err', mx[:6].round(3), typ_b[:6])
```