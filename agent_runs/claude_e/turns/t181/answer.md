Stay 1010006464 is a training stay, so RR is not only the tuning sel1 rows I assumed. That would also make indexing its 'prev' column into the fwd_2 states wrong, and the gate result from run 165 may be mislabelled.

I'll check which sources and stay-id ranges RR covers (its src column) and whether they match the fwd_2 states file, before redoing the gate analysis on the right rows.

```python
print(RR.src.value_counts().to_dict() if 'src' in RR else '')
pref = RR.stay.astype(str).str[:3]
print(pref.value_counts().to_dict())
print('fwd_2 stays prefix', pd.Series(z2['stay_id'].astype(str)).str[:3].value_counts().to_dict(), 'n stays', len(np.unique(z2['stay_id'])))
print('RR stays', RR.stay.nunique(), ' prev max', RR.prev.max(), ' len fwd_2 x', len(z2['x']))
for src_, grp in RR.groupby('src'):
    print(src_, grp.stay.nunique(), grp.prev.min(), grp.prev.max(), str(grp.stay.iloc[0]))
```