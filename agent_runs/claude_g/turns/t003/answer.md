The records have a regular structure. Readings fall on integer hours, and insulin records start 5 minutes later (hour h+0.0833) and run for one hour. So each dose is likely chosen right after the reading at hour h, and the next reading at h+1 sees about 55 minutes of that infusion. Insulin rates come in 0.5-unit steps (fine-bin edges 0.5 to 7). Norepinephrine dose bins saturate at 4.8 mg, i.e. a rate of 1.0 mcg/kg/min.

Next I want the basic structure:
- stay lengths and the timing of every event type within the hour;
- how often each drug is given, and the distributions of its rate and dose;
- most importantly, how each dose relates to the readings just before it (the dosing policy).

To identify the causal effect I need dose variation that the recorded history does not explain. If the rate is a deterministic function of the last reading, the data cannot separate dose from condition. If there is residual randomness, a regression adjusted for history can.

```python
pd.set_option('display.width', 250)
tr['frac'] = (tr.hour - np.floor(tr.hour)).round(4)
print(tr.groupby('event').frac.value_counts().groupby(level=0).head(5))
adm = tr[tr.event=='icu_admission'][['stay_id','end_hour']].set_index('stay_id').end_hour
print('ICU stay length (end_hour) quantiles:', adm.quantile([0,.05,.25,.5,.75,.95,1]).round(2).to_dict())
for drug in ['insulin','norepinephrine']:
    d = tr[tr.event==drug]
    print(drug, 'n=',len(d), 'stays with drug:', d.stay_id.nunique())
    print(' rate value counts top:', d.rate.round(3).value_counts().head(25).to_dict())
    print(' dose/rate ratio:', (d.dose/d.rate).describe().round(4).to_dict())
    print(' rate_unit', d.rate_unit.unique(), 'dose_unit', d.dose_unit.unique())
    print(' end-start:', (d.end_hour-d.hour).round(4).value_counts().head().to_dict())
```