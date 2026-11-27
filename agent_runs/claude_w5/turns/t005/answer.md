The normalisation tells me where edited doses land:
- **Insulin rate:** median 1.5, IQR 1.0. Doubling a typical 1.5 u/h rate moves z from 0 to 1.5; large rates clip at z = 10.
- **Norepinephrine rate:** median 0.47, IQR 0.59, and the recorded rate never exceeds 1.0. Several bin edges sit at 1.0, so any doubled rate above 1.0 lands in fine bin 16, which may never occur in training. The x2 norepinephrine edit therefore pushes the model out of its data range. That doesn't matter for the truth, but it does matter for how the edited model behaves.

Next I want the data-generating structure:
- per-stay event counts and stay lengths;
- the minute offsets of infusion records;
- whether map_arterial and map_cuff go together;
- how often rates change, and what the rate depends on (last glucose, previous rate).

This tells me whether dosing looks like a deterministic protocol of observed values plus noise, which would make adjustment for history identify the effect, or whether there is hidden confounding.

```python
tr['minute'] = np.round((tr.hour - np.floor(tr.hour))*60).astype(int)
print(tr.groupby('event').minute.value_counts().unstack(fill_value=0).iloc[:, :15].to_string())
g = tr.groupby('stay_id')
cnt = tr.pivot_table(index='stay_id', columns='event', values='hour', aggfunc='count', fill_value=0)
print(cnt.describe().T.to_string())
adm = tr[tr.event=='icu_admission'].set_index('stay_id').end_hour
print("ICU stay length:", adm.describe())
print("frac stays with insulin", (cnt.insulin>0).mean(), "with norepi", (cnt.norepinephrine>0).mean(), "with art", (cnt.map_arterial>0).mean())
print(pd.crosstab(cnt.insulin>0, cnt.norepinephrine>0))
```