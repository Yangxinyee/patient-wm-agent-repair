The first causal estimate is striking: insulin has no detectable effect on glucose_finger. The rate coefficient is +0.05 ± 0.04 mg/dL per u/h at 1 hour, and stays within ±0.15 at 2, 3, 4 and 6 hours. These are tight bounds: doubling insulin (about +2 u/h) would move glucose by under 0.3 mg/dL. Since m is randomised and measurement is unaffected, this points to a true insulin effect near zero.

Before I believe it, I want to rule out:
- an effect that differs by stratum and cancels in the pooled estimate;
- a nonlinear dose–response that a linear term misses;
- an effect visible only in glucose_lab, or only on the longer cumulative horizon that the readout spans (hours 13–24 after doses from hour 12).

So next I will:
1. Compare mean ΔG nonparametrically across m levels within each base stratum, at k = 1 and 3.
2. Regress glucose at hours 13–24 on the stay's cumulative random insulin since hour 12. That cumulative excess is still randomised, and it is the quantity behind the readout.

Flat ΔG across m levels in every stratum would confirm "no effect".

```python
for k in [1,3]:
    dd = d[d[f'gf_p{k}'].notna()].copy()
    dd['dG'] = dd[f'gf_p{k}'] - dd.glucose_finger
    dd['mq'] = pd.cut(dd.m, [0,0.6,0.9,1.1,1.4,1.8,4])
    # adjust for glucose within stratum by subtracting stratum-glucose-bin mean
    dd['gb'] = (dd.glucose_finger//10)
    dd['dGa'] = dd.dG - dd.groupby('gb').dG.transform('mean')
    t = dd.groupby(['base','mq'], observed=True).dGa.agg(['size','mean','sem']).round(2)
    print(f'k={k}'); print(t.unstack(0).to_string())
```