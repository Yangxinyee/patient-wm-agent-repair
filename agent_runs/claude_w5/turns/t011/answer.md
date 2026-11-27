Three facts about the measurements:
- **Glucose:** finger and lab readings agree on average (difference −0.04, sd 9.1), so both measure the same latent glucose with independent noise.
- **MAP:** cuff reads about 3 mmHg above arterial (sd of the difference 6.7).
- **Hourly change in finger glucose:** sd 14.3, and successive changes have autocorrelation −0.41. A large share of the variation is therefore measurement noise or fast mean reversion, not persistent change. That matters for the dynamics model I'll need.

Next I'll pin down the exact insulin policy at 5 mg/dL resolution: thresholds, base levels and the noise distribution. Then I'll test whether the rate residual (log rate minus the band mean) is exogenous:
- it should be uncorrelated with past glucose changes given the current reading;
- it should not persist within a stay, since a stay-level component would hint at a clinician or patient factor.

If the residual is independent noise, within-band rate variation identifies the causal effect.

```python
s = P[P.insulin.notna()].copy()
s['gb'] = (s.g_now//5)*5
t = s.groupby('gb').insulin.agg(['size','mean', lambda x: np.log(x).mean(), lambda x: np.log(x).std(), 'min','max'])
print(t.to_string())
# does finger vs lab matter for rate at hours with both? 
b = s[s.glucose_finger.notna() & s.glucose_lab.notna()]
for col in ['glucose_finger','glucose_lab']:
    bb = b.assign(gb=(b[col]//10)*10)
    print(col, bb.groupby('gb').insulin.apply(lambda x: np.log(x).mean()).round(2).to_dict())
```