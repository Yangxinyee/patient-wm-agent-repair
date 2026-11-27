Two findings.

**Insulin with no finger reading that hour.** These hours all have a lab glucose of at least 150. So the insulin rule reads whichever glucose was taken that hour, and lab readings may have their own threshold.

**Norepinephrine rule.**
- **Starting:** it starts when the last MAP is below about 65 (certain below 60).
- **Stopping:** once on, it mostly continues unless MAP is above 75.
- **Rate pattern:** the rate falls with MAP, but rises again at high MAP, with large spread (sd ≈0.3).

That non-monotone pattern suggests titration from the previous rate, not a fixed level per MAP band. Next I'll model the change in rate against current MAP (arterial where available) and the previous rate, looking for steps (discontinuities) and the noise structure.

```python
Non = N[(N.ne_on==1)&(N.ne_prev_on==1)].copy()
Non['m'] = Non.map_arterial.fillna(Non.map_cuff)
Non['src'] = np.where(Non.map_arterial.notna(),'art',np.where(Non.map_cuff.notna(),'cuff','none'))
Non['d'] = Non.norepinephrine - Non.ne_prev
Non['ratio'] = Non.norepinephrine / Non.ne_prev
print(Non.src.value_counts())
Non['mb'] = pd.cut(Non.m, [0,40,50,55,60,62,64,65,66,68,70,75,80,90,200])
print(Non.groupby(['src','mb']).agg(dmean=('d','mean'), dsd=('d','std'), lr=('ratio', lambda x: np.log(x).mean()), lrsd=('ratio', lambda x: np.log(x).std()), n=('d','size')).to_string())
```