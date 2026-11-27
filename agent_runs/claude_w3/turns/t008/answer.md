Three facts came out:
1. When finger and lab share an hour, the policy uses the finger reading (99.5% agreement; the remainder is the rounding at 150).
2. In lab-only hours it uses the lab value.
3. Insulin is never given in an hour without a fresh glucose reading, and such hours occur only when the last reading was 150 or below.

So patients with glucose above 150 are checked every hour, and the presence of a reading itself depends on glucose. The policy is: at each hour with a fresh reading (finger preferred), if it exceeds 150, give base(reading) × jitter.

Next I will finish the base-rate function, including the bands above 250, and study the jitter. I want to know whether it is independent across hours within a stay, independent of the trend in glucose (a change from the previous reading), independent of the stay's mean glucose, and independent of a simultaneous lab reading given the finger reading. Independence would make the jitter a clean natural experiment. Dependence on recorded quantities can be adjusted for. Dependence on a stay-level trait would be a red flag.

```python
q2 = pn[(pn.h>=0)&((pn.glucose_finger_age==0)|(pn.glucose_lab_age==0))].copy()
q2['pol'] = np.where(q2.glucose_finger_age==0, q2.glucose_finger, q2.glucose_lab)
on = q2[q2.pol>150].copy()
hb = pd.cut(on.pol, [150,180,220,250,260,270,280,290,300,320,340,400])
print(on.groupby(hb, observed=True).insulin.agg(['size','mean','median','std']).round(2).to_string())
```