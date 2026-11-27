Fitting one linear readout across all scaled doses costs precision. The rate readouts degrade to rms 0.12 for insulin and 0.06 for NE, with maxima near 0.8. The presence readouts stay excellent: insulin within 0.04, NE within 0.10, both far from the 0.5 gate threshold.

The presence gates turn the injection off wherever the drug is absent, so the rate readout only needs to be accurate where its drug is present. I'll refit each rate readout on drug-present rows only, keeping measurement bundles at zero so the carrier stays empty at most bundles. I'll report the mean bias per dose multiplier as well as the spread, because a bias would distort the ×0.5, ×2 and ×3 answers systematically.

```python
def fit_rate(nm, pres_nm, wmeas=10.0, lam=1e-6):
    rows = TT[pres_nm][:-len(Fn)] > 0.5
    X_ = np.r_[FF[:-len(Fn)][rows], Fn]; y_ = np.r_[TT[nm][:-len(Fn)][rows], np.zeros(len(Fn))]
    w_ = np.r_[np.ones(rows.sum()), wmeas*np.ones(len(Fn))]; s_ = np.sqrt(w_)
    b = np.linalg.solve((X_*s_[:,None]).T@(X_*s_[:,None]) + lam*np.eye(192), (X_*s_[:,None]).T@(y_*s_))
    return b
for nm, pn in [('tI','pI'), ('tN','pN')]:
    b = fit_rate(nm, pn); BB[nm+'_p'] = b
    e_all = FF@b - TT[nm]; off = 0; s_ = []
    for k in VAR:
        n_ = len(VAR[k]['F']); ek = e_all[off:off+n_]; pk = VAR[k][pn] > 0.5; off += n_
        s_.append('%s bias %+.3f rms %.3f' % (k, ek[pk].mean(), np.sqrt((ek[pk]**2).mean())))
    print(nm, '|b| %.0f' % np.linalg.norm(b), '| meas max %.3f' % np.abs(e_all[-len(Fn):]).max())
    print('   ', '; '.join(s_))
```