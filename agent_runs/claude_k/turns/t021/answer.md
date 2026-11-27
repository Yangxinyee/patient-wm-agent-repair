Next I'll run the cross-fitted partialling-out for both drugs at lags 1–8:
- norepinephrine: rows already on it at t-1; outcome arterial MAP at t+k; first stage the rate at t+j;
- insulin: rows with insulin running at t; outcome glucose_finger at t+k; first stage the later insulin rates.

If the insulin effects stay near -2.0, -2.25, -1.8… and norepinephrine's near +3.3, +5.4, +6.4…, my earlier estimates were right and the deconvolution can proceed. Big shifts in norepinephrine would mean its earlier shock was confounded through the policy inputs.

```python
def build(drug):
    rows=[]
    for t in range(1,24):
        j=col(t); f=hist_feats(t)
        f['stay']=np.arange(len(ids)); f['dose']= (ne if drug=='ne' else ins)[:,j]
        for k in range(1,9):
            jj=j+k
            f[f'y{k}'] = ((ma if drug=='ne' else gf)[:,jj]) if jj<=col(24) else np.nan
            f[f'd{k}'] = ((ne if drug=='ne' else ins)[:,jj]) if jj<=col(24) else np.nan
        rows.append(f)
    B = pd.concat(rows, ignore_index=True)
    B = B[(B.rp>0)] if drug=='ne' else B[(B.dose>0)]
    return B.reset_index(drop=True)
featcols = ['t','glast','gf0','gl0','gm1','gm2','mlast','ma0','mc0','mm1','mm2','ip','ipp','rp','rpp','has_art']
results = {}
for drug in ['ne','ins']:
    B = build(drug)
    X = B[featcols]
    rd = crossfit_resid(X, B.dose.values)
    out=[]
    for k in range(1,9):
        m = B[f'y{k}'].notna().values
        ry = crossfit_resid(X[m], B[f'y{k}'].values[m])
        b = (rd[m]@ry)/(rd[m]@rd[m]); e = ry - b*rd[m]; se = np.sqrt((e**2*rd[m]**2).sum())/(rd[m]@rd[m])
        md = B[f'd{k}'].notna().values
        rdk = crossfit_resid(X[md], B[f'd{k}'].values[md])
        fb = (rd[md]@rdk)/(rd[md]@rd[md])
        out.append((k, m.sum(), round(b,3), round(se,3), round(fb,4)))
    results[drug]=out
    print(drug, "resid sd of dose", rd.std().round(3), "of", B.dose.std().round(3))
    for o in out: print("  lag", o)
```