The insulin policy looks like a sliding scale on glucose.
- **When insulin is on:** it is always on when the last finger glucose is above 160. Below 140 it is almost never on, and between 140 and 160 it depends on whether insulin ran the previous hour.
- **Rate steps:** the rate rises in steps, about 1 u/h below 180, about 2 for 180–220, about 3 for 220–250, and higher above that.
- **Scatter within a step:** sd ≈0.37 at the 1 u/h level and ≈0.65 at the 2 u/h level, roughly proportional to the level. That suggests multiplicative noise.

That scatter is the variation that could identify the effect. But only if the rule depends on observed readings, not on unobserved true glucose.

Next I'll look at the exact rate distribution within a step, and at sharpness around the thresholds (finger glucose 175–185 and 215–225, in 1 mg/dL steps). A sharp jump at the recorded value means the rule uses the recorded reading. A blurred jump means it uses something unobserved, or a different reading.

```python
Qon = Q[Q.ins_on==1].copy()
print(Qon[(Qon.last_glucose_finger>160)&(Qon.last_glucose_finger<=180)].insulin.value_counts().sort_index().to_string())
print(Qon[(Qon.last_glucose_finger>180)&(Qon.last_glucose_finger<=220)].insulin.value_counts().sort_index().to_string())
# sharpness: mean rate by exact glucose value near 180 and 220 (only hours where glucose_finger measured at h)
cur = Qon[Qon.glucose_finger.notna()]
for lo,hi in [(170,190),(210,230),(240,260)]:
    s = cur[(cur.glucose_finger>=lo)&(cur.glucose_finger<=hi)].groupby('glucose_finger').insulin.agg(['mean','size'])
    print(s.T.to_string())
# when glucose_finger not measured at h: what drives rate? 
nocur = Qon[Qon.glucose_finger.isna()]
print("frac hours w/o current finger:", len(nocur)/len(Qon))
for lo,hi in [(170,190),(210,230)]:
    s = nocur[(nocur.last_glucose_finger>=lo)&(nocur.last_glucose_finger<=hi)].groupby('last_glucose_finger').insulin.agg(['mean','size'])
    print(s.T.to_string())
```