The run failed straight away: `Uc` calls `isnan`, so it needs NaN for an absent drug, not None. Nothing else ran.

I am rerunning the same check unchanged, except that absent drugs go to `Uc` as NaN and an unused helper is removed. The question is still whether the per-pattern readouts give the edited-minus-recorded dose change to within about 2% on the u's the evaluation will produce.

```python
import numpy as np, time
from sklearn.linear_model import Ridge
rId = np.asarray(RI_d, float); rNd = np.asarray(RN_d, float)
hasI_d = np.isfinite(rId) & (rId > 0); hasN_d = np.isfinite(rNd) & (rNd > 0)
mults = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
pairs_real = set()
for a, b, hi, hn in zip(rId, rNd, hasI_d, hasN_d):
    A = round(float(a), 4) if hi else None; Bn = round(float(b), 4) if hn else None
    for m in mults:
        if hi: pairs_real.add((round(A * m, 4), Bn))
        if hn: pairs_real.add((A, round(Bn * m, 4)))
pairs_real = sorted(pairs_real, key=lambda p: (p[0] is None, p[0] or 0, p[1] is None, p[1] or 0))
nn_ = lambda v: np.nan if v is None else v
t0_ = time.time()
UR = np.array([np.asarray(Uc(nn_(a), nn_(b)), float).ravel() for a, b in pairs_real])
key_d = [((round(float(a), 4) if hi else None), (round(float(b), 4) if hn else None)) for a, b, hi, hn in zip(rId, rNd, hasI_d, hasN_d)]
idx_map = {p: i for i, p in enumerate(pairs_real)}
UR_d = UR[[idx_map[k] for k in key_d]]
print('realistic pairs', len(pairs_real), 'in', round(time.time() - t0_, 1), 's; max |Uc - uk|', np.abs(UR_d - Uk).max().round(6))
aR = np.array([nn_(p[0]) for p in pairs_real], float); bR = np.array([nn_(p[1]) for p in pairs_real], float)
patR = np.where(~np.isnan(aR) & np.isnan(bR), 1, np.where(np.isnan(aR) & ~np.isnan(bR), 2, 3))
ro2 = {}
for nmr, k, yname in [('I_only', 1, 'a'), ('N_only', 2, 'b'), ('both_I', 3, 'a'), ('both_N', 3, 'b')]:
    yG = aG if yname == 'a' else bG; yRv = aR if yname == 'a' else bR
    X = np.vstack([UG[pat == k], UR[patR == k]]); y = np.r_[yG[pat == k], yRv[patR == k]]
    w = np.r_[np.full((pat == k).sum(), 0.05), np.ones((patR == k).sum())]
    m = Ridge(alpha=1e-8).fit(X, y, sample_weight=w); ro2[nmr] = m
    pR = m.predict(UR[patR == k]); pGg = m.predict(UG[pat == k])
    print(f'{nmr}: realistic n {(patR==k).sum()} rms {np.sqrt(((pR - yRv[patR==k])**2).mean()):.4f} max {np.abs(pR - yRv[patR==k]).max():.4f} | grid rms {np.sqrt(((pGg - yG[pat==k])**2).mean()):.4f}')
for drug_, ro_only, ro_both, base_r, has_self, has_other in [('ins', 'I_only', 'both_I', rId, hasI_d, hasN_d), ('ne', 'N_only', 'both_N', rNd, hasN_d, hasI_d)]:
    errs_ = {}
    for m in [0.0, 0.5, 2.0, 3.0]:
        d_true, d_est = [], []
        for i in np.where(has_self)[0]:
            a, b = key_d[i]
            pe = (round(a * m, 4), b) if drug_ == 'ins' else (a, round(b * m, 4))
            mdl = ro2[ro_both] if has_other[i] else ro2[ro_only]
            fe = mdl.predict(UR[[idx_map[pe]]])[0]; f1 = mdl.predict(UR[[idx_map[key_d[i]]]])[0]
            d_est.append(fe - f1); d_true.append((m - 1) * base_r[i])
        d_true = np.array(d_true); d_est = np.array(d_est)
        errs_[m] = (d_est.sum() / d_true.sum(), np.sqrt(((d_est - d_true)**2).mean()))
    print(drug_, 'readout-diff ratio est/true and rms:', {k: (round(v[0], 4), round(v[1], 4)) for k, v in errs_.items()})
```