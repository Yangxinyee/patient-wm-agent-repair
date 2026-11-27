Each single-drug pattern is read exactly by a linear map:
- insulin-only: rms 0.001 over rates 0 to 11.5;
- NE-only: rms 0.0001 over rates 0 to 3.

The combined pattern (both drugs in one bundle) is only approximately linear over the full grid: rms 0.058 for insulin and 0.035 for NE. Each of the four presence patterns is linearly separable from the others, with margins of about ±0.7 to ±1.

Combined bundles are the majority in the data: about 1,375 of the 2,319 drug bundles. So the readout must be accurate on combinations. What actually matters is the set of u's the evaluation will produce: each recorded (insulin, NE) pair, with one drug multiplied by 0, ½, 1.5, 2 or 3.

Next run:
- build that realistic set with my exact encode_action copy;
- refit the per-pattern readouts, weighting the realistic set heavily;
- report the error of the edited-minus-recorded readout difference, which is the quantity the injection is linear in.

Success would be difference errors below about 2% of the dose change for both drugs. That would make the gated, per-pattern surprise units accurate enough to build the z-clone edit.

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
t0_ = time.time()
UR = np.array([np.asarray(Uc(a, b), float).ravel() for a, b in pairs_real])
print('realistic pairs', len(pairs_real), 'u computed in', round(time.time() - t0_, 1), 's; check vs data u: ',
      end='')
key_d = [((round(float(a), 4) if hi else None), (round(float(b), 4) if hn else None)) for a, b, hi, hn in zip(rId, rNd, hasI_d, hasN_d)]
idx_map = {p: i for i, p in enumerate(pairs_real)}
UR_d = UR[[idx_map[k] for k in key_d]]
print('max |Uc - uk|', np.abs(UR_d - Uk).max().round(6))
aR = np.array([np.nan if p[0] is None else p[0] for p in pairs_real]); bR = np.array([np.nan if p[1] is None else p[1] for p in pairs_real])
patR = np.where(~np.isnan(aR) & np.isnan(bR), 1, np.where(np.isnan(aR) & ~np.isnan(bR), 2, 3))
ro2 = {}
for nmr, k, yname in [('I_only', 1, 'a'), ('N_only', 2, 'b'), ('both_I', 3, 'a'), ('both_N', 3, 'b')]:
    yG = aG if yname == 'a' else bG; yRv = aR if yname == 'a' else bR
    X = np.vstack([UG[pat == k], UR[patR == k]]); y = np.r_[yG[pat == k], yRv[patR == k]]
    w = np.r_[np.full((pat == k).sum(), 0.05), np.ones((patR == k).sum())]
    m = Ridge(alpha=1e-8).fit(X, y, sample_weight=w); ro2[nmr] = m
    pR = m.predict(UR[patR == k]); pGg = m.predict(UG[pat == k])
    print(f'{nmr}: realistic n {(patR==k).sum()} rms {np.sqrt(((pR - yRv[patR==k])**2).mean()):.4f} max {np.abs(pR - yRv[patR==k]).max():.4f} | grid rms {np.sqrt(((pGg - yG[pat==k])**2).mean()):.4f}')
def f_read(u, which):
    if which == 'I':
        return np.where(hasN_any(u), ro2['both_I'].predict(u), ro2['I_only'].predict(u))
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
    print(drug_, 'readout-difference ratio (est/true sum) and rms by multiplier:', {k: (round(v[0], 4), round(v[1], 4)) for k, v in errs_.items()})
```