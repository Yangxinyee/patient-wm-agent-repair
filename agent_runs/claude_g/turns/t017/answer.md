There is no discontinuity at the 150 mg/dL threshold: the jump in later glucose change is −0.29, +0.52, −0.02 and −0.08 mg/dL at 1, 2, 3 and 6 hours, against a smooth upward drift of about 10 mg/dL over 6 hours. So neither the amount of insulin (randomised multiplier) nor being on insulin at all (sharp threshold) changes glucose. **Insulin is inert in this ICU**, and the true insulin x2 and x1/2 readouts should be about 0.

For norepinephrine the margin effect is about 0 (−0.33/unit), but I have not tested an on/off effect. Below MAP 65 an infusion is always running. I will look at hours where norepinephrine was not running in the previous hour (ne_prev = 0):
- the start probability against MAP at h near 65, to see if there is a sharp start rule;
- the MAP change at 1–3 hours on either side of that threshold.

A jump would mean a saturating vasopressor effect: starting matters, doubling does not. No jump would mean norepinephrine is inert too.

```python
H2 = H.copy()
H2['started'] = (H2.ne>0) & (H2.ne_m1.fillna(0)==0)
off = H2[(H2.ne_m1.fillna(0)==0) & H2.map_arterial.notna() & H2.ne_m1.notna()].copy()
t = off.groupby(off.map_arterial.round()).agg(n=('ne','size'), pstart=('started','mean'), rate=('ne', lambda x: x[x>0].mean()))
print(t.loc[55:80].round(3).T.to_string())
for k in [1,2,3]:
    off['dM'] = off[f'ma_p{k}'] - off.map_arterial
    tt = off.groupby(off.map_arterial.round()).agg(n=('dM','count'), dM=('dM','mean'))
    lo = tt.loc[55:64]; hi = tt.loc[65:74]
    bl = np.polyfit(lo.index, lo.dM, 1, w=np.sqrt(lo.n)); bh = np.polyfit(hi.index, hi.dM, 1, w=np.sqrt(hi.n))
    print(f'k={k}: left(<65) limit {np.polyval(bl,64.5):.2f}, right(>=65) limit {np.polyval(bh,64.5):.2f}')
    print(tt.loc[55:75].dM.round(2).to_dict())
```