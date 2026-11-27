The run failed only because, inside a single band, the other band dummies are all-zero columns and make the design matrix singular. That is a coding bug, not a result. I will drop all-zero columns and rerun the same two checks: per-band effects and the glucose_finger measurement policy. What would confirm or refute linearity is the same as before.

```python
def surprise_reg(dd, ycol):
    s = dd[dd[ycol].notna()].copy()
    s['GF_m1f'] = s.GF_m1.fillna(s.GF); s['hasm1']=s.GF_m1.notna().astype(float)
    cols = [s[f'b{b}'].values for b in [1,2,3,4,6] if s[f'b{b}'].sum()>0] + [s.GF.values, s.GF_m1f.values, s.hasm1.values, s.u.values]
    X = np.column_stack(cols)
    yv = s[ycol].values
    b,*_ = np.linalg.lstsq(X, yv, rcond=None)
    r = yv - X@b
    cov = np.linalg.pinv(X.T@X)*(r@r)/(len(s)-X.shape[1])
    return b[-1], np.sqrt(cov[-1,-1]), len(s)
for bsel in [1,2,3]:
    sub = dd[dd.base==bsel]
    vals=[]
    for k in [1,2,3,4,6]:
        a,se,n = surprise_reg(sub, f'GF_p{k}')
        vals.append(f"k{k}:{a:.2f}±{se:.2f}")
    print("base",bsel, " ".join(vals))
P['GF_meas_next'] = g['GF'].shift(-1).notna().astype(int)
P['GL_meas_next'] = g['GL'].shift(-1).notna().astype(int)
z = P[(P.h>=0)&(P.h<=23)]
print(z.groupby(pd.cut(z.GF_ff,np.arange(60,320,10))).agg(n=('GF_meas_next','size'), pGFnext=('GF_meas_next','mean'), pGLnext=('GL_meas_next','mean')).to_string())
```