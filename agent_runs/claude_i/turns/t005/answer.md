Each stay-hour has at most one insulin and one norepinephrine record. Readings are irregular: finger glucose is present in 52–59% of hours and rises over time, lab glucose in 20%, arterial MAP in 46%, cuff MAP in 61%. Insulin is running in 31% of hours at hour 0 and 42% by hour 24; norepinephrine stays near 32%.

Next I'll look at a few stays row by row to see how dosing tracks the readings. In particular I want to see:
- whether insulin is titrated off the latest finger glucose;
- whether infusions run continuously once started;
- how norepinephrine follows MAP.

```python
def show(P, sid, cols=['gf','gl','ma','mc','ins','ne']):
    print(sid); print(P.loc[sid][cols].T.round(2).to_string())
ins_stays = P.groupby(level='stay_id').ins.count()
ne_stays = P.groupby(level='stay_id').ne.count()
print("insulin hours per stay dist:", ins_stays.value_counts().sort_index().to_dict())
print("NE hours per stay dist:", ne_stays.value_counts().sort_index().to_dict())
sids = P.index.get_level_values(0).unique()
for sid in sids[:6]:
    show(P,sid)
```