This is a striking result. The random part of the insulin dose has no detectable effect on later finger glucose, at any lag from 1 to 6 hours: about +0.01 ± 0.04 mg/dL per U/h at 1 hour, and never more than about 2 SE from zero. The residual is also unrelated to a same-hour lab glucose (coefficient −2e-5 ± 1.2e-4), so clinicians do not seem to dose on information hidden from the record.

My working hypothesis is that insulin rate does not move glucose in this simulated ICU. That would mean the true x2 and x1/2 effects on glucose are near zero.

I will not settle on that yet. Next I want to test whether the null result holds:
- for the response to the dose in bins, which catches a nonlinear dose-response;
- separately within each sliding-scale step, in case the effect depends on glucose level;
- for lab glucose as the outcome;
- at longer lags (up to 12 h);
- for the cumulative dose over several hours.

```python
# nonlinear: binned residual vs future glucose change (G_{h+k} - G_h), within step
mi2['rb'] = pd.cut(mi2.res, [-2,-0.5,-0.25,-0.1,0.1,0.25,0.5,2])
for k in [1,2,3,6]:
    mi2[f'dG_{k}'] = mi2[f'Gf_{k}'] - mi2.glucose_finger
    print(k, mi2.groupby('rb')[f'dG_{k}'].mean().round(2).to_dict())
# by step, k=2
for bb in [1,2,3,4,6]:
    d = mi2[(mi2.base==bb) & mi2['Gf_2'].notna()]
    X = np.column_stack([d.insulin.values, d.glucose_finger.values, d.glucose_finger.values**2/1e3])
    be, se = ols(d['Gf_2'].values, X)
    print("step", bb, "k=2 coef rate", round(be[1],3), "+-", round(se[1],3), len(d))
# lab glucose outcomes and longer lags
for k in [1,2,3,4,6,8,10,12]:
    xx[f'Gf_{k}'] = xx.groupby('stay_id').glucose_finger.shift(-k)
    xx[f'Gl_{k}'] = xx.groupby('stay_id').glucose_lab.shift(-k)
mi2 = xx[xx.insulin.notna() & xx.glucose_finger.notna()].copy()
mi2['base'] = base_rate(mi2.glucose_finger.values); mi2 = mi2[mi2.base>0]
mi2['res'] = np.log(mi2.insulin) - np.log(mi2.base)
for k in [1,2,3,4,6,8,10,12]:
    for nm in ['Gf','Gl']:
        d = mi2[mi2[f'{nm}_{k}'].notna()]
        X = np.column_stack([d.insulin.values, ctrl(d)])
        be, se = ols(d[f'{nm}_{k}'].values, X)
        print(f"{nm} k={k} n={len(d)} coef rate {be[1]:.3f} +- {se[1]:.3f}")
```