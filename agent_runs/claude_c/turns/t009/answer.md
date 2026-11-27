When both readings exist, the finger-stick decides the tier: doses follow the finger tier even when the lab tier differs. When only a lab value exists, the lab value decides, with somewhat lower adherence (92%). So the decision is driven by recorded readings.

Next I will characterise the residual randomness:
- the distribution of the rate within each tier;
- whether the deviation from the tier's base rate, and whether a dose is skipped, relate to pre-treatment variables: glucose within the tier, previous glucose, previous rate, MAP, norepinephrine, hour, and whether a lab value exists.

If they show no relationship, the noise works as a randomised experiment, and I can estimate the insulin effect by comparing next-hour glucose across noise levels at the same reading.

```python
P['tf'] = tier(P.glucose_finger)
P['base'] = P.tf
P['dev'] = P.insulin - P.tf
P['map_any'] = P.map_arterial.fillna(P.map_cuff)
P['ne_prev'] = P.groupby(level=0).norepinephrine.shift(1)
dd = P[P.glucose_finger.notna() & (P.tf>=1) & (P.glucose_finger.round()!=150)&(P.glucose_finger.round()!=180)&(P.glucose_finger.round()!=220)&(P.glucose_finger.round()!=260)].copy()
for t in [1,2,3,4]:
    x = dd[dd.tf==t]
    print("tier",t, x.insulin.value_counts(dropna=False).sort_index().to_dict())
dd['skip'] = dd.insulin.isna().astype(float)
x = dd[dd.insulin.notna()].copy()
x['hh'] = x.index.get_level_values('h')
import statsmodels.api as sm
```