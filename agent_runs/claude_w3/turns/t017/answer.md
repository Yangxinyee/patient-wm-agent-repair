The norepinephrine policy is a titration with memory, not a memoryless rule like insulin's:
- It uses the arterial MAP when present, else the cuff MAP.
- It is always on at MAP ≤ 62. Between 62 and 90 the chance of being on falls, and it tracks whether norepinephrine ran the previous hour almost exactly.
- It is often continued in hours with no fresh MAP (25% on).
- The mean rate is U-shaped in MAP: high at low MAP, lowest around 70–75, and higher again above 85. That fits a rate that is carried forward and adjusted, with weaning lagging behind a rising MAP.

To use the same identification idea, I need the titration rule: how the change in rate from the previous hour depends on the current MAP and the previous rate, how starts and stops happen, and whether the leftover noise is random. Next I will tabulate:
- the starting probability and starting rate when the drug was off, against fresh MAP;
- the rate change when it was on, against fresh MAP and previous rate;
- behaviour in hours without a fresh MAP.

If a clean rule plus independent noise emerges, the noise is my instrument. If the adjustments depend on unrecorded state, norepinephrine effects will be harder to pin down.

```python
pn0['map_pol'] = np.where(pn0.map_arterial.notna(), pn0.map_arterial, pn0.map_cuff)
x = pn0[(pn0.h>=1)].copy()
x['ne'] = x.norepinephrine; x['dr'] = x.ne - x.ne_prev
mb = pd.cut(x.map_pol, [0,50,55,60,62,65,70,75,80,85,90,200])
# starts
st0 = x[(x.ne_prev==0)&x.map_pol.notna()]
print("STARTS (prev off, fresh MAP):")
print(st0.groupby(mb.reindex(st0.index), observed=True).agg(n=('ne','size'), p_start=('ne', lambda v:(v>0).mean()),
      rate=('ne', lambda v: v[v>0].mean()), sd=('ne', lambda v: v[v>0].std())).round(3).T.to_string())
# continuing
c1 = x[(x.ne_prev>0)&x.map_pol.notna()]
print("ON (prev on, fresh MAP): change in rate")
print(c1.groupby(mb.reindex(c1.index), observed=True).agg(n=('ne','size'), p_stop=('ne', lambda v:(v==0).mean()),
      dr=('dr', lambda v: v[c1.loc[v.index,'ne']>0].mean()), sd=('dr', lambda v: v[c1.loc[v.index,'ne']>0].std()),
      ratio=('ne', lambda v: (v[v>0]/c1.loc[v[v>0].index,'ne_prev']).mean())).round(3).T.to_string())
# no fresh MAP
c2 = x[(x.ne_prev>0)&x.map_pol.isna()]
print("prev on, no fresh MAP: n", len(c2), " p_stop", (c2.ne==0).mean().round(3), " frac unchanged", (c2.dr==0).mean().round(3), " mean dr", c2.dr.mean().round(4))
c3 = x[(x.ne_prev==0)&x.map_pol.isna()]
print("prev off, no fresh MAP: n", len(c3), " p_start", (c3.ne>0).mean().round(4))
# examples of rate sequences
ex = x[x.stay_id==x[x.ne>0].stay_id.iloc[100]][['h','map_arterial','map_cuff','ne_prev','ne']]
print(ex.to_string())
```