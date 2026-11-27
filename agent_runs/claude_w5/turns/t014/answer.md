The insulin noise behaves like pure exogenous randomisation. Its correlation is essentially zero (|r| ≤ 0.013) with:
- the previous hour's noise;
- the past glucose change;
- lab minus finger glucose (a proxy for hidden true glucose);
- MAP, norepinephrine and the hour.

The spread of stay-level average noise (0.080) is close to what pure chance gives (0.075), so there is no hidden stay-level dosing style.

Conclusion: given the recorded glucose, the insulin rate is randomised. Regressing later glucose on the rate, with flexible control for the reading that set the base rate, identifies the causal effect.

Next I'll record this in the notebook and estimate the insulin effect directly. For lags k = 1…6 I'll regress finger glucose at t + k on rate_t, controlling for the current reading (fixed effects per integer glucose value, which absorb the base rate) and recent history. This gives the total effect of a one-hour rate change, including the policy's later feedback. A clearly negative coefficient that grows over a few hours would mean insulin really lowers glucose; a coefficient near zero would mean no effect.

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
import statsmodels.api as sm if False else None
```