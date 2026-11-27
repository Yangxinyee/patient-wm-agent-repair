Both cross-effects are negligible. Insulin has no effect on MAP (-0.02 ± 0.02 at every lag). Norepinephrine gives a marginal +0.6 ± 0.25 on glucose at lag 1 and nothing after. So each drug acts essentially only on its own reading: insulin on glucose, norepinephrine on MAP.

The readouts run to hour 24, so a dose at hour 12 can matter up to lag 12. Next I'll extend the double-ML total effects and first stages to lag 12 for both drugs. I'll fit parametric impulse responses (β·(a^k − r^k)/(a − r), which covers insulin's k·0.6^(k−1) form) directly to the total effects through the first-stage convolution, and compare with the free deconvolution. A two-pole fit that tracks all 12 lags within their standard errors would give smooth, trustworthy tails. Systematic misfit at long lags would suggest a slower component, such as a permanent effect.

```python
def build12(drug):
    rows=[]
    for t in range(1,24):
        j=col(t); f=hist_feats(t)
        f['stay']=np.arange(len(ids)); f['dose']= (ne if drug=='ne' else ins)[:,j]
        for k in range(1,13):
            jj=j+k
            f[f'y{k}'] = ((ma if drug=='ne' else gbest)[:,jj]) if jj<=col(24) else np.nan
            f[f'd{k}'] = ((ne if drug=='ne' else ins)[:,jj]) if jj<=col(24) else np.nan
        rows.append(f)
    B = pd.concat(rows, ignore_index=True)
    B = B[(B.rp>0)] if drug=='ne' else B[(B.dose>0)]
    return B.reset_index(drop=True)
res12 = {}
for drug in ['ins','ne']:
    B = build12(drug); X = B[featcols]; rd = crossfit_resid(X, B.dose.values)
    tot=[]; se_=[]; fs=[]
    for k in range(1,13):
        m = B[f'y{k}'].notna().values; ry = crossfit_resid(X[m], B[f'y{k}'].values[m])
        b = (rd[m]@ry)/(rd[m]@rd[m]); e = ry-b*rd[m]; se = np.sqrt((e**2*rd[m]**2).sum())/(rd[m]@rd[m])
        md = B[f'd{k}'].notna().values; rdk = crossfit_resid(X[md], B[f'd{k}'].values[md])
        tot.append(b); se_.append(se); fs.append((rd[md]@rdk)/(rd[md]@rd[md]))
    res12[drug] = (np.array(tot), np.array(se_), np.array(fs))
    print(drug, "tot", np.round(tot,2)); print("   se", np.round(se_,2)); print("   f", np.round(fs,3))
    print("   h", np.round(deconv(tot, fs),2))
```