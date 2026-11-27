My last run failed on a stray, invalid import line, so nothing ran and the notebook entry was not written. I'll rerun the same step without it, using numpy least squares for the regressions.

The plan is unchanged. For lags k = 1…8 I'll regress finger glucose at t + k on rate_t, with fixed effects for each integer glucose value at t (which absorb the base rate) and the previous reading as a control. Since the dose noise is randomised, a negative coefficient means insulin lowers glucose. How it changes across lags shows the time course, including the policy's later feedback.

```python
with open('/work/notes.md','a') as f:
    f.write("""
## Runs 2-13: data structure
- All stays 25.17h; readings on the hour (-1..24), drug records at h+5min (0..24), each runs 1h.
- Norepi dose = 4.8*rate (fixed weight), rate capped at 1.0. Insulin dose = rate. Rates in 0.5 steps.
- INSULIN POLICY (sliding scale on recorded glucose at same hour, finger else lab): g<150 none; 150-180 base 1; 180-220 base 2;
  220-260 base 3; 260-300 base 4; >300 ~6. rate = round0.5(base*exp(eps)), eps sd~0.31.
  Thresholds sharp at recorded value (180 exactly = 50/50) -> policy uses measured value, not latent.
- eps is exogenous: corr with eps_prev, past dg, lab-finger, MAP, norepi, hour all ~0; no stay-level component.
  => within-band dose variation is randomised; regression of future glucose on rate | g_now identifies causal effect.
- finger vs lab: no bias, sd diff 9.1. cuff = art + 3 (sd 6.7). hourly finger diff sd 14.3, lag1 autocorr of diffs -0.41.
""")
s = s.sort_values(['stay_id','h']).reset_index(drop=True)
gs = s.groupby('stay_id')
for k in range(1,9):
    s[f'f_lead{k}'] = gs.glucose_finger.shift(-k)
def ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X@b
    XtX_inv = np.linalg.pinv(X.T@X)
    se = np.sqrt(np.diag(XtX_inv) * (r@r)/(len(y)-X.shape[1]))
    return b, se
base_mask = s.insulin.notna() & s.glucose_finger.notna() & s.g_prev.notna()
for k in range(1,9):
    d = s[base_mask & s[f'f_lead{k}'].notna()]
    gfe = pd.get_dummies(d.glucose_finger.astype(int)).values.astype(float)
    X = np.column_stack([d.insulin.values, d.g_prev.values, gfe])
    b, se = ols(d[f'f_lead{k}'].values, X)
    print(k, len(d), "coef rate:", round(b[0],3), "+-", round(se[0],3))
```