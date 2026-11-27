X3 matches the raw tool states exactly, and its z block equals zm3/ZZt, so the target is right. The mismatch must therefore come from ctx or from e/u. The error was the same for every atom offset, which points to ctx, or to `atoms_by_bundle` being indexed differently from the 5156 state rows; it has 5158 keys. This run checks three things: that sid3/hr3 equal the npz stay/hour, the key range and an excerpt of `atoms_by_bundle`, and enc2.py, to recall what eu2 expects.

```python
print('sid match',bool((np.asarray(sid3)==S['stay_id']).all()),' hour match',bool(np.allclose(np.asarray(hr3,dtype=float),S['hour'])))
ks_=sorted(atoms_by_bundle.keys()); print('abb keys',ks_[:3],ks_[-3:],type(ks_[0]))
print('abb[1] excerpt:',str(atoms_by_bundle[1])[:300])
print(open('/work/enc2.py').read()[:3300])
```