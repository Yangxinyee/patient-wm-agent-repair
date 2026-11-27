The regression failed for a mechanical reason, not a substantive one. I required a MAP reading at hour s, so the "no MAP" band indicator is always zero and the design matrix is singular. I will drop that column and switch the helper to a pseudo-inverse so this cannot recur. The test itself is unchanged: future MAP on the titration innovation, controlling for the history, at lags 1–12 h.

```python
def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X]).astype(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r_ = y - X@beta
    s2 = r_@r_/(len(y)-X.shape[1])
    cov = s2*np.linalg.pinv(X.T@X)
    return beta, np.sqrt(np.diag(cov))
for k in [1,2,3,4,6,8,12]:
    for nm, base in [('Ma','map_arterial'),('Mc','map_cuff')]:
        dd = r[r[f'{nm}_{k}'].notna() & r[base].notna()]
        X = np.column_stack([dd.innov, dd[base], dd[base]**2/100, dd.lrate_prev, dd.band==1, dd.band==2])
        be,se = ols(dd[f'{nm}_{k}'].values, X)
        print(f"{nm} k={k} n={len(dd)} coef innov {be[1]:.3f} +- {se[1]:.3f}   mean rate {dd.norepinephrine.mean():.3f}")
```