Norepinephrine is dosed differently from insulin:
- Rates are in 0.01 steps and change gradually: hour-to-hour steps have sd 0.145 and mean +0.012.
- About 94% of records have a MAP reading at the same hour, so 6% are given without a fresh reading.
- Norepinephrine starts when MAP is around 62, perhaps a threshold near 65.

It could be a titration rule (previous rate plus a term in MAP below target, plus noise) or a level rule. Next I'll regress the rate on the previous rate and on current MAP, arterial or cuff, with an indicator for a missing reading. I'll also inspect the residuals to find the exogenous component. If the rate is the previous rate plus a function of MAP plus independent noise, that noise is the instrument, as it was for insulin.

```python
ne = ne[ne.norepinephrine_prev.notna()].copy()
ne['map_src'] = np.where(ne.map_arterial.notna(), 'art', np.where(ne.map_cuff.notna(), 'cuff', 'none'))
for src in ['art','cuff','none']:
    d = ne[ne.map_src==src]
    if src!='none':
        mb = pd.cut(d.map_now, [0,50,55,60,62,64,65,66,68,70,75,80,90,200])
        print(src, d.groupby(mb).agg(n=('h','size'), dr=('dr','mean'), dr_sd=('dr','std'), lr_ratio=('norepinephrine', 'mean')).round(3).to_string())
    else:
        print(src, len(d), d.dr.mean(), d.dr.std())
```