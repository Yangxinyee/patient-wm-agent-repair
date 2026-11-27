The panel covers readings at hours −1 to 24 and drug records at hours 0 to 24. The number of insulin records per hour rises from about 9,400 to 12,600 as more stays start insulin; norepinephrine records slowly fall from about 9,500 to 9,200.

Next, the insulin dosing policy:
- At an insulin record at hour h, is a glucose_finger reading present at the same hour?
- How does the rate depend on that glucose, the previous rate and glucose_lab?
- How large and what shape is the residual noise?
- What triggers starting insulin (the first glucose above a threshold?) and stopping it?

The same questions apply to norepinephrine and MAP.

```python
P = P.sort_values(['stay_id','h']).reset_index(drop=True)
grp = P.groupby('stay_id')
for c in ['glucose_finger','glucose_lab','map_arterial','map_cuff','insulin','norepinephrine']:
    P[c+'_prev'] = grp[c].shift(1)
    P[c+'_next'] = grp[c].shift(-1)
ins = P[P.insulin.notna()]
print("insulin records:", len(ins))
print("has finger same h:", ins.glucose_finger.notna().mean(), " lab same h:", ins.glucose_lab.notna().mean(),
      " either:", (ins.glucose_finger.notna()|ins.glucose_lab.notna()).mean())
print("prev insulin present:", ins.insulin_prev.notna().mean())
print("rate value counts:", ins.insulin.value_counts().sort_index().to_string())
# first insulin record per stay: what glucose
first = ins.groupby('stay_id').head(1)
print("first insulin hour dist:", first.h.describe())
print("glucose at first insulin:", first.glucose_finger.describe())
```