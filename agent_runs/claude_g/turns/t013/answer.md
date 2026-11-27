Whether glucose is measured at h+1 does not depend on the insulin multiplier (95–96% in every quintile), so a regression on hours with a reading at h+1 is not biased by selection. statsmodels is unavailable, so I will write a small OLS with cluster-robust standard errors in numpy.

Now the main test. For k = 1, 2, 3, 4 and 6, I regress G(h+k) − G(h) on the insulin rate at h, with fixed effects for the base stratum, a spline in G(h), and the previous rate. Because m is randomised, the rate coefficient is the causal effect of one extra u/h at h on glucose k hours later. For k > 1 this is the total effect, including the clinicians' later reaction to the changed glucose. A negative coefficient at k = 1 means insulin lowers glucose. A coefficient near 0 means the effect is small or absent.

```python
def ols(y, X, cluster=None):
    X = np.asarray(X, float); y = np.asarray(y, float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    r = y - X @ b
    if cluster is None:
        S = (X * r[:,None]**2).T @ X
    else:
        cl = pd.factorize(cluster)[0]
        Xr = X * r[:,None]
        g = np.zeros((cl.max()+1, X.shape[1])); np.add.at(g, cl, Xr)
        S = g.T @ g
    V = XtX_inv @ S @ XtX_inv
    return b, np.sqrt(np.diag(V)), r
def design(d, extra=[]):
    cols = [np.ones(len(d))]
    for bv in [2,3,4,6]: cols.append((d.base==bv).astype(float).values)
    gz = (d.glucose_finger.values-200)/50
    for kn in [-1,-0.4,0.4,1.2]: cols.append(np.maximum(gz-kn,0))
    cols.append(gz)
    cols.append(d.ins_m1.fillna(0).values)
    for e in extra: cols.append(e)
    return np.column_stack(cols)
stay = d.index.get_level_values('stay_id').values
for k in [1,2,3,4,6]:
    sel = d[f'gf_p{k}'].notna().values
    dd = d[sel]
    X = design(dd, [dd.ins.values])
    y = (dd[f'gf_p{k}'] - dd.glucose_finger).values
    b, se, _ = ols(y, X, stay[sel])
    print(f'k={k}: n={sel.sum()}, coef on rate_h = {b[-1]:.3f} (se {se[-1]:.3f})')
```