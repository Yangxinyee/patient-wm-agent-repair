The panel shows each measurement type at a roughly constant share of hours. Finger glucose appears at 53–62% of hours, rising over the stay, and insulin records at 34–46%, also rising. The other types are flat: lab glucose 20%, arterial MAP 46%, cuff MAP 61%, norepinephrine 33%.

Next I will examine:
1. Whether measurement patterns are a property of the stay (e.g., only some stays have arterial lines or finger sticks).
2. How the insulin rate relates to the latest glucose readings. That is the dosing policy: how deterministic it is and which readings it uses.

If the rate is nearly a deterministic function of the visible readings, there is little "as-if random" variation. I would then need other identifying variation, such as between-stay protocol differences or discontinuities.

```python
st = PN.groupby(level='stay_id').apply(lambda x: x.notna().any())
print("per-stay presence fractions:\n", st.mean())
print(pd.crosstab(st.map_arterial, st.glucose_finger))
print(pd.crosstab(st.insulin, st.glucose_finger))
print(pd.crosstab(st.norepinephrine, st.map_arterial))
# Insulin vs glucose in same hour
x = PN.reset_index()
m = x[x.insulin.notna() & x.glucose_finger.notna()]
print("corr insulin vs finger glucose same hour:", m[['insulin','glucose_finger']].corr().iloc[0,1])
m2 = x[x.insulin.notna()]
print("insulin rows with finger glucose same hour:", x.loc[x.insulin.notna(),'glucose_finger'].notna().mean())
print("insulin rows with lab glucose same hour:", x.loc[x.insulin.notna(),'glucose_lab'].notna().mean())
# binned relationship
m = m.copy(); m['gb'] = pd.cut(m.glucose_finger, [0,100,140,180,220,260,300,350,500])
print(m.groupby('gb').insulin.describe().to_string())
```