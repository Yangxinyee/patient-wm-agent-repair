Norepinephrine starts when MAP is at or below about 65 mmHg. The start probability is 54% in the 64–65 band and drops off with a small tail up to 75 mmHg. That tail suggests the trigger reading is sometimes a different MAP than the one I assumed (arterial, else cuff).

Once running, norepinephrine almost always continues while MAP ≤ 75 and is stopped only part of the time above 75. It also runs in 26% of hours with no MAP reading, so continuation depends on the stay's state, not on a fresh reading.

Next I will model the titration. I will regress the change in log rate (or the rate ratio) on the current MAP and the previous rate, then examine the residual:
- its spread;
- its autocorrelation within a stay;
- its relation to other readings.

That tells me whether norepinephrine has an as-if-random dose component like insulin, which would identify its effect on MAP.

```python
r = ne[(ne.h>=0) & ne.norepinephrine.notna() & ne.ne_prev.notna()].copy()
r['ratio'] = r.norepinephrine/r.ne_prev
r['lratio'] = np.log(r.ratio)
print(r.lratio.describe())
print(r.groupby(pd.cut(r.Mtrig,[0,45,50,55,60,62,64,65,66,68,70,72,75,80,90,200])).lratio.agg(['mean','std','count']).round(3).to_string())
print("no MAP:", r[r.Mtrig.isna()].lratio.agg(['mean','std','count']))
# relation to previous rate level
print(r.groupby(pd.cut(r.ne_prev,[0,0.05,0.1,0.2,0.4,0.6,0.8,0.95,1.0])).lratio.agg(['mean','std','count']).round(3).to_string())
```