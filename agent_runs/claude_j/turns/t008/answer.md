The sliding scale is exact. Insulin is off at a recorded glucose_finger of 150 or below. Base rates are 1 for 150-180, 2 for 180-220, 3 for 220-260, 4 for 260-300 and 6 above 300. At exactly 150, 180, 220 and so on, about half the hours fall on each side, consistent with a threshold on the unrounded value. Within a band the rate is the base rate times right-skewed noise, rounded to 0.5. When glucose_finger is missing, insulin is almost never given (2.4%). Glucose_finger is never missing when the last reading exceeded 160, so on insulin it is measured every hour.

This supports two independent identification strategies:
- **Regression discontinuity:** at each threshold the dose jumps by about one unit while the patient is essentially the same.
- **Within-band noise:** if the multiplier is exogenous, regressing the next reading on the rate within a band identifies the effect.

Next I will run a fuzzy RD for the one-hour effect: the jump in next-hour glucose_finger divided by the jump in insulin rate, at each threshold. If the RD estimates are consistent across thresholds, I have a clean one-step causal effect to compare with the within-band estimate later. If they vary wildly, the effect is nonlinear or there is more structure.

```python
P['GF_next'] = g['GF'].shift(-1)
P['GL_next'] = g['GL'].shift(-1)
P['I0'] = P.I.fillna(0.0)
def rd(df, T, w=15, y='GF_next'):
    s = df[(df.GF>=T-w)&(df.GF<=T+w)&(df.GF!=T)&df[y].notna()].copy()
    s['x'] = s.GF - T; s['above'] = (s.x>0).astype(float)
    X = np.column_stack([np.ones(len(s)), s.above, s.x, s.x*s.above])
    out = {}
    for col in [y,'I0']:
        b, res, *_ = np.linalg.lstsq(X, s[col].values, rcond=None)
        r = s[col].values - X@b
        cov = np.linalg.inv(X.T@X)*(r@r)/(len(s)-4)
        out[col] = (b[1], np.sqrt(cov[1,1]))
    return len(s), out
d = P[(P.h>=0)&(P.h<=23)]
for T in [150,180,220,260,300]:
    for w in [10,20]:
        n,o = rd(d,T,w)
        jy, sy = o['GF_next']; ji, si = o['I0']
        print(f"T={T} w={w} n={n} jumpGFnext={jy:.2f}±{sy:.2f} jumpI={ji:.3f}±{si:.3f} effect/unit={jy/ji:.2f}")
```