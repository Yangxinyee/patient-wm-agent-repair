The records load fine: 2.06 million events over 30,000 stays (27k events in tuning). Glucose readings are mostly finger sticks (453k) plus lab draws (156k). There are 306k insulin and 247k norepinephrine one-hour records. Insulin rates come in 0.5-unit steps, typically 1–3 U/h with a few very high; norepinephrine is capped at 1.0 mcg/kg/min.

Events seem to fall on whole hours, at least in the first stay. Next I will check:
- whether times are integer hours;
- how long stays are;
- how many stays receive each drug, and from when;
- one stay with insulin, printed in full, to see how doses track glucose.

This tells me how much within-stay dose variation exists to work with.

```python
print("hour fractional part counts:", (tr.hour % 1 != 0).sum(), "of", len(tr))
print(tr.hour.describe())
stay_len = tr[tr.event=='icu_admission'].set_index('stay_id').end_hour
print("ICU stay length (end_hour of icu_admission):"); print(stay_len.describe())
g = tr.groupby('stay_id')
has_ins = g.apply(lambda d: (d.event=='insulin').any())
has_ne = g.apply(lambda d: (d.event=='norepinephrine').any())
print("frac stays with insulin", has_ins.mean(), "with NE", has_ne.mean(), "both", (has_ins&has_ne).mean())
ins = tr[tr.event=='insulin']
ne = tr[tr.event=='norepinephrine']
print("insulin start hour of first record:"); print(ins.groupby('stay_id').hour.min().describe())
print("NE first record:"); print(ne.groupby('stay_id').hour.min().describe())
print("insulin rate value counts:"); print(ins.rate.value_counts().sort_index().head(40))
sid = ins.stay_id.iloc[0]
d = tr[tr.stay_id==sid]
print(d.to_string())
```