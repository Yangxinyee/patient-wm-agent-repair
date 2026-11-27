The states file holds 4,356 bundles from the 100 tuning stays. Each bundle has 992 numbers: `out` (192), the four h slots (768), then the z posterior mean in the last 32 columns. It also records the stay, the hour and insulin/norepinephrine flags.

The factual file gives every reading's observed value and the model's predicted mean, so surprise = observed − predicted can be matched to reading bundles by stay and hour.

Next I'll regress the 32-d z posterior at reading bundles on the surprises of glucose_finger, glucose_lab, map_arterial and map_cuff. I'm looking for the z-direction g that encodes "glucose came out higher than predicted", and its MAP counterpart.

| Result | Meaning |
|---|---|
| Surprises explain much of z (high R² along a clear direction) | g exists; the remapping lever is worth a forward test |
| Surprises explain little | z does not carry reading information this way, and I drop the idea |

```python
X3 = z3['x']; sid3 = z3['stay_id']; hr3 = z3['hour']; hi3 = z3['has_insulin']; hn3 = z3['has_norepinephrine']
Zp = X3[:, 960:992]
print('hour frac sample', np.unique(np.round(hr3 % 1, 4))[:10])
fa2['sur'] = fa2['observed'] - fa2['predicted_mean']
piv = fa2.pivot_table(index=['stay_id','hour'], columns='event', values='sur', aggfunc='first')
key = pd.MultiIndex.from_arrays([sid3, np.round(hr3, 4)])
piv.index = pd.MultiIndex.from_arrays([piv.index.get_level_values(0), np.round(piv.index.get_level_values(1).astype(float), 4)])
J = piv.reindex(key)
rd = (~J.isna().all(axis=1)).values & (hi3 == 0) & (hn3 == 0)
print('reading bundles', rd.sum(), 'of', len(rd))
evs = ['glucose_finger','glucose_lab','map_arterial','map_cuff']
F = np.column_stack([np.nan_to_num(J[e].values / (15.0 if 'glucose' in e else 6.0)) for e in evs] + [(~J[e].isna()).values.astype(float) for e in evs])
Fr = np.column_stack([F[rd], np.ones(rd.sum())]); Zr = Zp[rd]
coef_, *_ = np.linalg.lstsq(Fr, Zr, rcond=None)
res_ = Zr - Fr @ coef_
r2 = 1 - res_.var(0) / Zr.var(0)
print('per-dim R2 (top):', sorted([(round(r2[i],3), i) for i in range(32)], reverse=True)[:10])
print('total R2', 1 - res_.var(0).sum() / Zr.var(0).sum())
for j, e in enumerate(evs):
    gvec = coef_[j]; print(e, 'coef norm', round(np.linalg.norm(gvec),3), 'top dims', np.argsort(-np.abs(gvec))[:5])
```