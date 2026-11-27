The norepinephrine policy differs from insulin's. Its probability falls smoothly from about 98% at MAP 40 to near 0 at MAP 90, with the steep part between 64 and 72. The rate given is higher at lower MAP (0.77 at 40, 0.39 at 64, about 0.34 above 66). The previous rate tracks the current rate almost exactly, which points to hour-to-hour titration with a lot of carry-over rather than a sliding scale. Arterial and cuff readings give similar curves, so either reading probably feeds the decision. The crash was only a name clash: `P.ne` is the pandas "not equal" method, so I must write `P['ne']`.

Next I will model the titration rule. I want to see how rate_t relates to rate_{t−1} and the current MAP (ratios and differences, and whether a stopped infusion restarts). I also want NE use in hours with no MAP reading. The titration form matters because a sticky rule makes past doses a proxy for persistent hidden blood-pressure trouble, which would confound naive regressions.

```python
nomap = P[P.map_arterial.isna() & P.map_cuff.isna() & (P.index.get_level_values(1)>=0)]
print("hours no MAP:", len(nomap), " frac NE:", (nomap['ne']>0).mean().round(3), " frac NE prev>0:", (nomap['ne_prev']>0).mean().round(3))
R = P[P.map_any.notna() & (P.index.get_level_values(1)>=0)].copy()
R['m4'] = (R.map_any//4*4).clip(40,96).astype(int)
R['pbin'] = pd.cut(R.ne_prev, [-0.01,0.0001,0.1,0.3,0.6,0.99,1.0])
print(R.groupby(['m4','pbin'], observed=True)['ne'].mean().unstack().round(3).to_string())
R2 = R[(R.ne_prev>0)&(R['ne']>0)]
R2 = R2.assign(lr=np.log(R2['ne']/R2.ne_prev))
print(R2.groupby('m4').lr.agg(['size','mean','std']).round(3).to_string())
```