The surprise-based estimates are about five times more precise than the RD and agree with it. The fixed-dose impulse response of one extra unit/h for one hour is:

| hours after | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| change (mg/dL) | -2.00 | -2.38 | -2.08 | -1.70 | -1.16 | -0.91 | -0.65 | -0.49 | -0.25 | -0.11 |

The cumulative effect is about -11.8 mg/dL. This curve is almost exactly -2·k·0.6^(k-1), a gamma-like kernel from two first-order stages with rate 0.6 per hour, whose total is -2/(0.4)² = -12.5.

Before relying on it, two checks:
1. **Linearity and homogeneity:** is the effect the same across dose bands? The evaluation multiplies every dose, so heterogeneity weighted by dose size matters.
2. **Selection:** glucose_finger can go unmeasured when glucose drops, and insulin lowers glucose.

Next I will estimate the surprise effects separately by band and examine the measurement policy, that is, P(glucose_finger measured at h+1 | last glucose_finger). Similar per-unit effects across bands would confirm linearity. Measurement depending on the level would mean the long-horizon estimates need a selection-robust check.

```python
for bsel in [1,2,3]:
    sub = dd[dd.base==bsel]
    vals=[]
    for k in [1,2,3,4,6]:
        a,se,n = surprise_reg(sub, f'GF_p{k}')
        vals.append(f"k{k}:{a:.2f}±{se:.2f}")
    print("base",bsel, " ".join(vals))
# measurement policy for GF
P['GF_meas_next'] = g['GF'].shift(-1).notna().astype(int)
P['GL_meas_next'] = g['GL'].shift(-1).notna().astype(int)
z = P[(P.h>=0)&(P.h<=23)]
print(z.groupby(pd.cut(z.GF_ff,np.arange(60,320,10))).agg(n=('GF_meas_next','size'), pGFnext=('GF_meas_next','mean'), pGLnext=('GL_meas_next','mean')).to_string())
# hours since last GF measurement, for low glucose
```