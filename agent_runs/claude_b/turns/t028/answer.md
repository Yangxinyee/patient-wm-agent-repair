The fit failed because, within a single reading type, the "is arterial" indicator is all ones and duplicates the intercept. I will add that column only when pooling both types, and make the regression helper fall back to a pseudo-inverse when the matrix is singular. The question is unchanged: is there a robust step in the next MAP change at 65 and 75 that matches the norepinephrine jump there?

```python
def ols(y, X, cl=None, names=None):
    X = np.asarray(X, float); y = np.asarray(y, float)
    XtX = X.T@X; iX = np.linalg.pinv(XtX); b = iX@(X.T@y); e = y - X@b
    if cl is None:
        V = iX * (e@e)/(len(y)-X.shape[1])
    else:
        cl = pd.factorize(np.asarray(cl))[0]
        S = np.zeros((cl.max()+1, X.shape[1])); np.add.at(S, cl, X*e[:,None])
        V = iX @ (S.T@S) @ iX
    return b, np.sqrt(np.maximum(np.diag(V),0))
def rd_ne2(S, c, h, ycol, pooled=False):
    S = S[(S.x>=c-h)&(S.x<=c+h)&(S.x!=c)&S[ycol].notna()]
    x = S.x.values-c; a=(x>0).astype(float); n1=S.ne_p1.values
    cols = [np.ones(len(S)), a, x, a*x, n1, n1*a, n1*x]
    if pooled: cols.append((S.xtype=='art').astype(float).values)
    X = np.column_stack(cols)
    by,sey = ols(S[ycol].values, X, S.stay.values); bd,sed = ols(S['ne'].values, X, S.stay.values)
    return by[1],sey[1],bd[1],len(S)
for typ in ['art','cuff','all']:
    S0 = C if typ=='all' else C[C.xtype==typ]
    for c in [65,75]:
        for h in [5,8,12]:
            jy,se,jd,n = rd_ne2(S0, c, h, 'y1', pooled=(typ=='all'))
            print(f"{typ:4s} c={c} h={h:2d}: jump y1 {jy:+.3f}+-{se:.3f}  jump NE {jd:+.4f}  per-unit {jy/jd:+.2f}+-{se/abs(jd):.2f}  n={n}")
```