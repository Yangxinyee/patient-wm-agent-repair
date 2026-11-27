Each row of the file is one (stay, hour, edit) with the model's shift, its standard error over the 20 paths, the factual predicted mean, and a "reached" flag. There are 844 insulin rows for 100 stays, not 1,200, so rows seem to exist only at certain hours. My guess is the hours where the record has a finger glucose reading (arterial MAP for norepinephrine). "Reached" would then mean the hour comes after the first edited dose.

One stay already shows the model's insulin ×2 effect growing steadily to −9 mg/dL by hour 24, so per-stay patterns vary. Next I'll match the rows to the tuning records to confirm both rules, so my true-effect calculation can use exactly the same stay-hours.

```python
tuP = Ptu.reset_index()
for edit, col, dcol in [('insulin_x2','glucose_finger','insulin'),('ne_x2','map_arterial','norepinephrine')]:
    sub = de[de.edit==edit][['stay_id','hour','reached']].copy()
    rec = tuP[tuP.stay_id.isin(setA)&(tuP.h>=13)&(tuP.h<=24)][['stay_id','h',col,dcol]].rename(columns={'h':'hour'})
    m = rec.merge(sub, on=['stay_id','hour'], how='left')
    m['row'] = m.reached.notna()
    print(edit, pd.crosstab(m[col].notna(), m.row))
    # reached rule: first drug time >=12 before hour
    first = tuP[tuP.stay_id.isin(setA)&(tuP.h>=12)&tuP[dcol].notna()].groupby('stay_id').h.min()
    m['first'] = m.stay_id.map(first)
    m['pred_reached'] = m['first'].notna() & (m.hour > m['first'])
    mm = m[m.row]
    print(pd.crosstab(mm.reached, mm.pred_reached))
```