The omission puzzle resolves on the timing. Of about 9,400 omitted hours, only 371 had a finger reading the hour before. So omissions are almost all first-trigger hours after a gap without a reading. Their next-hour change distribution is only slightly lower than for given hours (medians −1 vs +1 in base 1, −11 vs −5 in base 3). Since my estimates use the rate noise among given doses and the readouts scale recorded doses, omission matters only as a minor confounder.

The remaining issue for the dynamic (multi-hour) estimates is measurement selection. Whether glucose is measured at hour t seems to depend on whether insulin ran at t−1, which earlier doses influence. To correct with inverse-probability weights, I need the measurement rule.

Next I will model when glucose_finger is measured, as a function of insulin at t−1, the last reading, hours since the last reading, and norepinephrine. If it depends only on recorded history, the readings are missing at random and weighting or a Kalman-filter model handles it.

```python
c['meas'] = c.gf.notna()
gg = c.groupby(level='stay_id')
c['last_gf_any'] = gg.gf.transform(lambda x: x.ffill().shift(1))
# hours since last gf
def hrs_since(x):
    out=np.full(len(x),np.nan); last=None
    for i,v in enumerate(x.values):
        out[i] = (i-last) if last is not None else np.nan
        if not np.isnan(v): last=i
    return pd.Series(out,index=x.index)
c['hsince'] = gg.gf.transform(hrs_since)
c['ins_prev_given'] = (c.ins_prev>0)
m = c[c.index.get_level_values('h')>=1]
print(m.groupby('ins_prev_given').meas.mean())
mm = m[~m.ins_prev_given]
print(mm.groupby([pd.cut(mm.last_gf_any,[0,100,120,140,150,160,180,250]), mm.hsince.clip(upper=5)],observed=True).meas.mean().unstack().round(3).to_string())
mi = m[m.ins_prev_given]
print(mi.groupby([pd.cut(mi.last_gf_any,[0,140,150,160,180,250,400])],observed=True).meas.agg(['mean','size']).round(3).to_string())
print(m.groupby(m.nor_prev>0).meas.mean())
```