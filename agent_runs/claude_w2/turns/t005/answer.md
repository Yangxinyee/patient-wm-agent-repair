Stays are short and regular: about 24 hours, with readings on the hour and infusion records starting 5 minutes past the hour (for example 0.0833). Each infusion is an hourly record, so in effect clinicians pick a rate every hour. Glucose_finger and map_cuff are very frequent, while glucose_lab is rarer.

Next I'll describe the timing and the dosing policy:
- the fractional part of the hour for each event type;
- how many stays get each drug, and for how many hours;
- the rate distributions;
- whether the rate depends on the latest reading (confounding by indication).

This shapes the causal estimation strategy. For example, if rates are chosen from the last reading plus random variation, I can use the random part as a natural experiment.

```python
tr['frac'] = (tr.hour - np.floor(tr.hour)).round(4)
print(tr.groupby('event').frac.value_counts().groupby(level=0).head(5))
for drug in ['insulin','norepinephrine']:
    d = tr[tr.event==drug]
    print(drug, 'stays with drug:', d.stay_id.nunique(), 'records per stay:', d.groupby('stay_id').size().describe().to_dict())
    print(d.rate.value_counts().sort_index().head(40))
    print((d.dose - d.rate).abs().max(), d.rate_unit.unique(), d.dose_unit.unique())
```