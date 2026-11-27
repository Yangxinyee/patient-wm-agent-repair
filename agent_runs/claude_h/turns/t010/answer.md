Patients on insulin always get a finger reading the next hour. Off insulin, a finger reading is taken 31.5% of the time. statsmodels is not installed, so I will write my own least squares with stay-clustered standard errors in numpy.

Next I will run the planned comparison of the next-hour finger-glucose change on the current insulin rate:
- naive, with no controls;
- with fine glucose-bin fixed effects (5 mg/dL bins) and the previous rate;
- the same, plus the previous glucose change and stay-hour.

If adding controls moves the coefficient strongly negative, the naive association is confounded by the sliding scale. If the coefficient stays stable once the current reading is controlled, the within-bin jitter looks exogenous.

```python
def ols(y, X, cluster=None, names=None):
    X = np.asarray(X, float); y = np.asarray(y, float)
    XtX = X.T@X; b = np.linalg.solve(XtX, X.T@y); r = y - X@b
    XtXi = np.linalg.inv(XtX)
    if cluster is not None:
        cl = pd.factorize(np.asarray(cluster))[0]
        S = np.zeros((X.shape[1],)*2)
        U = np.zeros((cl.max()+1, X.shape[1]))
        np.add.at(U, cl, X*r[:,None])
        S = U.T@U
        V = XtXi@S@XtXi
    else:
        V = XtXi*(r@r)/(len(y)-X.shape[1])
    se = np.sqrt(np.diag(V))
    return b, se, r
def fe(series, width):
    return pd.get_dummies((series//width).astype(int), prefix='b', drop_first=True).values.astype(float)
d = P[P.gf.notna() & P.gf_next.notna()].copy().reset_index()
d['dG'] = d.gf_next - d.gf
y = d.dG.values; cl = d.stay_id.values
X1 = np.c_[np.ones(len(d)), d.ins]
b,se,_ = ols(y, X1, cl); print("naive dG~ins:", b[1].round(3), se[1].round(3))
X2 = np.c_[d.ins, d.ins_prev.fillna(0), fe(d.gf.clip(40,400), 5), np.ones(len(d))]
b,se,_ = ols(y, X2, cl); print("ctrl G-bins + ins_prev: ins", b[0].round(3), se[0].round(3), " ins_prev", b[1].round(3), se[1].round(3))
d['dGprev'] = (d.gf - d.gf_prev)
m = d.dGprev.notna()
X3 = np.c_[d.ins, d.ins_prev.fillna(0), d.dGprev.fillna(0), fe(d.gf.clip(40,400), 5), d.ih, np.ones(len(d))][m.values]
b,se,_ = ols(y[m.values], X3, cl[m.values]); print("ctrl + dGprev + hour: ins", b[0].round(3), se[0].round(3), "ins_prev", b[1].round(3), "dGprev", b[2].round(3))
print("n", len(d), m.sum())
```