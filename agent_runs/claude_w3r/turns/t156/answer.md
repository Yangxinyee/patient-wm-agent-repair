E holds the rates only weakly: grouped-CV R² is 0.64 for the insulin rate and 0.49 for the norepinephrine rate. The direct "rank-1 `b_in` on E" plan is therefore too noisy, because the huge h dominates E after `prelude_norm`.

But u (the action-encoder output) is a deterministic function of the bundle's drug start atoms, so it should carry the rates without noise. Next I'll probe both rates linearly from u. I'll also probe them from the prelude attention head-2 value vector (W_V2·norm1(u)), which is what head 2 copies into the slot tokens. R² near 0.99 would mean I can inject a clean, dedicated rate component there before the noise from h enters.

```python
Wpre = {k: load(k) for k in NAMES if k.startswith('core.prelude.0.')}
print({k: tuple(v.shape) for k, v in Wpre.items()})
UA = []
for r in recE:
    B_, rows_, EU_ = stay_data(r['stay']); UA.append(np.asarray(EU_[r['k']][1], dtype=np.float64).ravel())
UA = np.stack(UA); print('u shape', UA.shape)
for nm, hk in [('ri','has_i'),('rn','has_n')]:
    yv = np.array([r[nm] for r in recE]); msk = np.array([r[hk] for r in recE])
    Xf = np.column_stack([UA, np.array([r['has_i'] for r in recE]), np.array([r['has_n'] for r in recE])])
    p = cvp(RidgeCV(alphas=np.logspace(-6,2,17)), Xf[msk], yv[msk], groups=gr[msk], cv=GroupKFold(5))
    print(nm, 'u-probe R2', round(1-np.var(yv[msk]-p)/np.var(yv[msk]),4), 'resid sd', round(np.std(yv[msk]-p),4))
```