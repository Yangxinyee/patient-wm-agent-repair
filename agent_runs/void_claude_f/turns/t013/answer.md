The dose scatter behaves like an independent hourly coin toss. Residuals of log rate around the sliding-scale base have:
- sd 0.33;
- lag-1 autocorrelation 0.005 within stays;
- essentially no between-stay component (variance of stay means 0.0064, which is about what sampling noise alone gives);
- no trend with hour or step.

So given the triggering reading, the recorded rate is the scale's base times a random factor. That is a natural randomised experiment for the dose, unless the factor secretly tracks the true glucose beyond the reading.

Next I will check that caveat and get a first estimate of the effect:
1. Does the residual correlate with a same-hour lab glucose, given the finger reading? A correlation would mean clinicians see more than the record shows.
2. Regress next-hour and later glucose changes on the residual (or log rate), controlling for the current reading.

A clearly negative coefficient on future glucose with no lab correlation would confirm a real, identifiable insulin effect. A coefficient near zero would suggest insulin does little in this simulated ICU.

```python
xx = xx.sort_values(['stay_id','h'])
# future finger glucose at h+k
for k in range(1,7):
    xx[f'Gf_{k}'] = xx.groupby('stay_id').glucose_finger.shift(-k)
    xx[f'Gl_{k}'] = xx.groupby('stay_id').glucose_lab.shift(-k)
mi2 = xx[xx.insulin.notna() & xx.glucose_finger.notna()].copy()
mi2['base'] = base_rate(mi2.glucose_finger.values)
mi2 = mi2[mi2.base>0]
mi2['res'] = np.log(mi2.insulin) - np.log(mi2.base)
# 1. residual vs lab glucose at same hour given finger
b = mi2[mi2.glucose_lab.notna()]
import statsmodels.api as smapi
```