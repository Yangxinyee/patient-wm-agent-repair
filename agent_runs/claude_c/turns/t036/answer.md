The group-by failed only because `st.ne` resolves to the DataFrame "not equal" method; I will use bracket indexing. The states also show the bundle structure:
- Drug bundles at h + 0.083 contain the new infusion atoms plus END atoms for the previous hour's infusions. For example, hour 1.0833 has two atoms for a single NE record.
- The 100 rows at .1667 are the stay-end bundles at 25.17 h.

Next I will tabulate the bundle types, check whether "has_insulin" also marks bundles that only end an insulin record, and join the states to the panel (dose, reading, tier). Then I will run the z-on-dose regression.

```python
st['frac'] = (st.hour - np.floor(st.hour)).round(4)
print(st.groupby([st['ins'], st['ne'], st['frac']]).size())
st['h'] = np.floor(st.hour).astype(int)
# join recorded doses at that hour
pt = Ptu.reset_index()
pt = pt.rename(columns={'h':'h'})
st2 = st.merge(pt[['stay_id','h','insulin','norepinephrine','glucose_finger','glucose_lab','map_arterial','map_cuff']], on=['stay_id','h'], how='left')
dr = st2[st2.frac==0.0833]
print(pd.crosstab(dr['ins'], dr.insulin.notna()), pd.crosstab(dr['ne'], dr.norepinephrine.notna()))
```