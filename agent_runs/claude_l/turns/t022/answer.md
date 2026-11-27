The CSV has one row per stay, hour and edit, with the shift, its standard error over paths, the factual predicted mean and a "reached" flag. Stay 2020000000's insulin rows start at hour 15 and are unreached there. Two things are still unclear: whether rows exist only at hours with a recorded reading of the outcome code, and whether "reached" means an insulin record started at or after 12 and before that hour.

I'll check this against the tuning record for a few stays. Then I'll reproduce the summary means from the CSV, testing pooled over reached rows against per-stay averages, so that my true-effect estimates use exactly the evaluation's definition.

```python
tu['ih'] = np.floor(tu.hour).astype(int)
for s in [2020000000, 2020000016, 2020000007]:
    r = tu[tu.stay_id==s]
    print(s, "finger hrs>=13:", sorted(r[(r.event=='glucose_finger')&(r.ih>=13)].ih.tolist()))
    print("   insulin hrs:", sorted(r[r.event=='insulin'].ih.tolist()))
    print("   art hrs>=13:", sorted(r[(r.event=='map_arterial')&(r.ih>=13)].ih.tolist()), " NE hrs:", sorted(r[r.event=='norepinephrine'].ih.tolist()))
    print("   csv ins rows:", de[(de.stay_id==s)&(de.edit=='insulin_x2')][['hour','reached']].values.tolist())
    print("   csv ne rows:", de[(de.stay_id==s)&(de.edit=='ne_x2')][['hour','reached']].values.tolist())
for e in de.edit.unique():
    x = de[(de.edit==e)&de.reached]
    print(e, "pooled:", x['shift'].mean().round(3), "per-stay avg:", x.groupby('stay_id')['shift'].mean().mean().round(3), len(x), x.stay_id.nunique())
```