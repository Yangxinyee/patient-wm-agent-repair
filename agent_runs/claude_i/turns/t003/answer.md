Units are uniform: insulin is always in units/hour and units, norepinephrine always in mcg/kg/min and mg. So there is no unit-mixing trap. In the first stay, readings fall on whole hours and insulin records start 5 minutes later (hour + 0.0833), each lasting 1 hour. That means a dose sits in its own bundle just after the readings it probably responds to.

Next I'll look at the structure of the records:
- stay lengths;
- which stays get each drug, and when;
- distributions of rates and doses;
- the dose/rate ratio for norepinephrine, which in mg vs mcg/kg/min would reveal patient weight;
- the timing of events within the hour.

```python
pd.set_option('display.width',250)
adm = train[train.event=='icu_admission'][['stay_id','end_hour']].set_index('stay_id').end_hour
print("stay length (h):", adm.describe())
ins = train[train.event=='insulin']; ne = train[train.event=='norepinephrine']
print("stays w insulin", ins.stay_id.nunique(), " w NE", ne.stay_id.nunique())
print("insulin rate dist:\n", ins.rate.describe(), "\n", ins.rate.value_counts().head(30))
print("NE rate dist:\n", ne.rate.describe(), "\n", ne.rate.round(3).value_counts().head(30))
print("insulin dose/rate:", (ins.dose/ins.rate).describe())
print("NE dose/rate:", (ne.dose/ne.rate).describe())
print("insulin end-start:", (ins.end_hour-ins.hour).describe())
for ev in ['glucose_finger','glucose_lab','map_arterial','map_cuff','insulin','norepinephrine']:
    d=train[train.event==ev]
    frac=(d.hour*60).round().mod(60)
    print(ev, "minute-of-hour top:", frac.value_counts().head(5).to_dict(), " hour range", d.hour.min(), d.hour.max())
```