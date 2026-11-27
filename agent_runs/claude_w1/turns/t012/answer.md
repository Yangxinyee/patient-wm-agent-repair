statsmodels isn't installed. The notes and the panel columns were written before the error, so nothing is lost. I'll do the regressions with numpy least squares and robust standard errors.

This first look at glucose dynamics regresses next-hour finger glucose on the current insulin rate, within bins of current finger glucose. The bins absorb the clinician's rule, so the slope should mostly reflect the random dose noise.
- A negative slope, stable across bins, suggests insulin lowers glucose.
- A near-zero slope suggests little effect.
- Slopes that change with the glucose level would point to a nonlinear effect, such as one proportional to glucose.

```python
def ols(y, X, names=None, cluster=None):
    X = np.asarray(X, float); y = np.asarray(y, float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    if cluster is None:
        meat = (X * e[:,None]).T @ (X * e[:,None])
    else:
        df = pd.DataFrame(X * e[:,None]); df['c'] = np.asarray(cluster)
        S = df.groupby('c').sum().values
        meat = S.T @ S
    V = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    if names is None: names = [f"x{i}" for i in range(X.shape[1])]
    return pd.DataFrame({'b':b,'se':se}, index=names), e

rows=[]
for gb, sub in A.groupby('gb', observed=True):
    if len(sub)<200: continue
    X = np.column_stack([np.ones(len(sub)), sub.glucose_finger, sub.ins0])
    r,_ = ols(sub.g_next, X, ['c','g','ins'])
    rows.append((str(gb), len(sub), r.loc['g','b'], r.loc['ins','b'], r.loc['ins','se'], sub.ins0.std()))
print(pd.DataFrame(rows, columns=['gbin','n','b_g','b_ins','se_ins','sd_ins']).to_string())
```