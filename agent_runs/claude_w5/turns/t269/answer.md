A small change to the prelude's u token alone explains most of the discrepancy. A δu of norm 0.06, about 0.5% of |u|, cuts the j3 error from 2.1e-3 to 4.6e-4. At j5, a δu of norm 0.015 brings it to 8e-5. The e token cannot explain j5 at all, since the slots don't attend to it there.

Since z, which takes u directly, is exact, the tool seems to feed the prelude an action token that differs slightly from the u given to z_post. Possible reasons are an extra embedding on action atoms, or some action-specific term.

To identify it, I'll fit δu_tok separately at several drug bundles (j3, j5, j7, j9, 11, 13) and compare them, both their cosines and their direction relative to the norm1-scaled u.
- If the δu vectors are nearly the same at every bundle, the tool adds a constant term to the action token. I can estimate it, add it locally and confirm the error drops to about 1e-6.
- If they vary with the doses, the tool's action encoding differs in a dose-dependent way, for example in the numeric features or bins.

```python
SOL = {}
for j in [3,5,7,9,11,13]:
    o, s_ = fit_tok(j, 'du_tok'); SOL[j] = s_
    print(j, [ (a['code'][:3], a['kind'], a.get('rate')) for a in B[j][1]], o[-2:])
keys = list(SOL)
M_ = np.stack([SOL[k] for k in keys]); nrm = np.linalg.norm(M_,axis=1,keepdims=True)
print('norms', nrm.ravel().round(4)); print('cos matrix'); print((M_/nrm @ (M_/nrm).T).round(3))
# relation to u itself
for k in keys:
    dk = prep(k, Xo5); uu = dk['u'].numpy()
    print(k, 'cos(du,u)', float(SOL[k]@uu/np.linalg.norm(SOL[k])/np.linalg.norm(uu)).__round__(3))
```