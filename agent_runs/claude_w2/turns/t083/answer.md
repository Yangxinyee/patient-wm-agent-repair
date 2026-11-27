The records do not support the simple story.

**What the one-step regression showed** (glucose_finger at t, n = 228,280, residual sd 12.9):
- **Glucose persistence.** The lag coefficients are 0.48, 0.30 and 0.21, summing to 0.98. Read through noisy readings, glucose is almost fully persistent.
- **Insulin coefficients by lag.** They are strongly negative at lags 1–3 (−2.0, −1.5, −0.9). They turn positive at lags 4–9 (+0.2, +0.8, +0.65, +0.4, +0.3, +0.2), and are about 0 at lag 10. So, given recent readings, an older dose predicts glucose rising back as its effect wears off.
- **Consistency with K.** The implied impulse response (−2.0, −2.4, −2.6, −2.2, −1.5, …) matches my deconvolution K over the first 5 lags. Its tail is heavier, because an AR(3) on noisy readings is only approximate.

**Conclusion.** The correct one-step model needs dose memory over about 10 hours, with a sign change. A correction on lags 1–2 alone, carried forward by persistence near 1, would overshoot badly.

**What matters now** is how far the model's own one-step dose coefficients fall short at each lag. If the model already captures the late positive terms, only the short lags need fixing. I fit the same regression to the model's teacher-forced predicted means, using fwd_4 (100 train stays, no new forward pass). For comparison I also fit it to the observed values in the same rows.

```python
f4g = f4[f4.event == 'glucose_finger'][['stay_id','hour','predicted_mean','observed']].rename(columns={'hour':'h'})
f4g['h'] = f4g['h'].round().astype(int)
Dm = b0.merge(f4g, on=['stay_id','h'], how='inner')
print('merged', len(Dm), 'obs==gf share', np.mean(np.isclose(Dm.observed, Dm.gf)))
Dm = Dm[Dm[['G1','G2','G3']].notna().all(1)]
def ols(df, ycol, cols):
    Xo = np.column_stack([np.ones(len(df))] + [df[c_].values for c_ in cols])
    b_, *_ = np.linalg.lstsq(Xo, df[ycol].values, rcond=None)
    r_ = df[ycol].values - Xo @ b_
    se_ = np.sqrt(np.diag(np.linalg.pinv(Xo.T @ Xo)) * r_.var())
    return b_, se_, r_.std()
bm_, sem_, sdm_ = ols(Dm, 'predicted_mean', cols_)
bo_, seo_, sdo_ = ols(Dm, 'gf', cols_)
print('n', len(Dm), 'resid sd model', round(sdm_, 2), 'obs', round(sdo_, 2))
print(f"{'':6s} {'model':>14s} {'obs(same rows)':>16s} {'obs(all train)':>15s}")
for i_, nm_ in enumerate(['const'] + cols_):
    print(f"{nm_:6s} {bm_[i_]:7.2f}+-{sem_[i_]:.2f} {bo_[i_]:8.2f}+-{seo_[i_]:.2f} {beta[i_]:10.2f}")
```