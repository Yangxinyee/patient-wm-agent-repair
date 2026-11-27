The records are regular: readings at whole hours, then an insulin record about 5 minutes later, with rate equal to dose since each record lasts one hour. Insulin rates come in 0.5-unit steps.

Next I'll map the structure before estimating any effect:
- timing patterns (fractional hours);
- stay lengths;
- which stays receive which drug and which MAP type;
- how doses relate to the latest reading.

The dosing policy is what decides whether the effect is identifiable. I'm hoping to find dose variation not explained by the observed history, since that would allow a clean regression.

```python
tr['frac'] = (tr.hour % 1).round(4)
print(tr.groupby('event').frac.value_counts().groupby(level=0).head(5).to_string())
st = tr.groupby('stay_id')
ends = tr[tr.event=='icu_admission'].set_index('stay_id').end_hour
print("ICU stay length (h):", ends.describe().to_string())
has = tr.groupby(['stay_id','event']).size().unstack(fill_value=0)
print((has>0).mean().to_string())
print(pd.crosstab(has.insulin>0, has.norepinephrine>0))
print(pd.crosstab(has.map_arterial>0, has.map_cuff>0))
print(pd.crosstab(has.map_arterial>0, has.norepinephrine>0))
print(pd.crosstab(has.glucose_finger>0, has.insulin>0))
print(tr[tr.event=='insulin'].rate.value_counts().sort_index().to_string())
print(tr[tr.event=='norepinephrine'].rate.describe())
print(tr[tr.event=='norepinephrine'].rate.value_counts().sort_index().head(60).to_string())
```