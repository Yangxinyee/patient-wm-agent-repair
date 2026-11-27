Run 2 shows the record layout:
- Readings sit on integer hours.
- Infusion records start 5 minutes after the hour (hour + 0.0833) and run for one hour, so dose equals rate × 1 h.
- About 276k insulin records and 242k norepinephrine records.

Next I'll characterise the timing, rate values, stay lengths and how often each drug is given. Then I'll look at how the dose relates to the most recent readings, i.e. the clinicians' policy. To estimate causal effects later I need to know whether doses are a deterministic function of recorded readings or contain independent variation. If doses are a noisy function of observed readings, conditioning on history may identify the effect. If they track something unrecorded, I'll need another strategy.

```python
d = tr.copy()
d['minute'] = np.round((d.hour - np.floor(d.hour))*60).astype(int)
print(d.groupby('event').minute.value_counts().unstack(fill_value=0).iloc[:, :12])
adm = d[d.event=='icu_admission']
print("stay length (end_hour):"); print(adm.end_hour.describe())
for ev in ['insulin','norepinephrine']:
    x = d[d.event==ev]
    print(ev, 'n stays with', x.stay_id.nunique(), 'rates value counts top:')
    print(x.rate.round(3).value_counts().head(25))
    print(x.rate.describe())
    print('dose==rate?', np.allclose(x.rate, x.dose), x.rate_unit.unique(), x.dose_unit.unique())
for ev in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
    x = d[d.event==ev]
    print(ev, x.value.describe().round(2).to_dict(), 'stays', x.stay_id.nunique())
```