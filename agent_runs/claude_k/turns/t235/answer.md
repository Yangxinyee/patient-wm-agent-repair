Run 218 failed: no drug bundle shares a stay-hour with any reading, and the lag matrix came out singular. So the drug-bundle times in the states file are not on the same hour grid as the readings in `R`. They may be fractional or offset, or the stay-id keys may not match.

Before redoing the regression, I'll print, for one stay, the drug-bundle hours and the reading hours next to each other. I'll also count how many readings get any lagged drug bundle under my lag definition. If the drug bundles turn out to sit at fractional hours, I'll measure lags in continuous hours instead.

```python
s0_=int(R.stay_id.values[0])
m0_=(stA==s0_)
print('S2 hours (first 30) for stay', s0_, np.round(hrA[m0_][:30],3))
print('drug flags', (dIA|dNA)[m0_][:30].astype(int))
print('reading hours', np.round(R.hour.values[R.stay_id.values==s0_][:20],3))
print('types', type(stA[0]), type(R.stay_id.values[0]), 'bys keys', len(bys), list(bys.keys())[:3])
```