The starting dose of norepinephrine is not random. It depends strongly on history:

- **Current MAP.** About 0.55 mcg/kg/min at MAP 40, versus 0.09 at MAP 64.
- **Previous MAP.** Log dose falls 0.059 per mmHg of previous MAP, so patients whose pressure was already low get more.
- **Hour of stay.** Log dose falls 0.055 per hour.

Controlling for the previous MAP cuts the start-dose association from −9.8 to −1.9 ± 0.4 mmHg per unit, and the hour dependence suggests more hidden structure. This looks like a hidden severity that the clinician sees: sicker patients get larger starting doses and their MAP keeps falling. Because titration multiplies the starting dose forward, the whole norepinephrine level carries that severity signal.

The titration surprise, by contrast, is unrelated to past MAP and shows no causal effect. So my current belief is a true norepinephrine effect of about 0, with the dose level being a confounded severity marker. A model trained on records could learn "more norepinephrine means MAP falls", which would make a ×2 edit predict a spurious MAP drop.

Next I will check whether the original model does this: per-stay norepinephrine shifts, split by whether the infusion started after hour 12. I will also check how variable the insulin shifts are. Negative ×2 shifts concentrated in post-12 starters would show the model is exploiting the confound.

```python
f = pd.read_csv('/work/fwd_1/dose_edit_original.csv')
f = f[f.reached]
ps = f.groupby(['stay_id','edit'])['shift'].mean().unstack()
# NE start info per tuning stay
Pt['stay']=Pt.index.get_level_values(0); Pt['hr']=Pt.index.get_level_values(1)
ne_before = Pt[(Pt.hr<12)].groupby('stay')['ne'].apply(lambda x:(x>0).any())
ne_after = Pt[(Pt.hr>=12)].groupby('stay')['ne'].apply(lambda x:(x>0).any())
ps['ne_before12'] = ne_before.reindex(ps.index); ps['ne_after12']=ne_after.reindex(ps.index)
print(ps[['ne_x2','ne_half']].describe().round(3))
print(ps.groupby('ne_before12')[['ne_x2','ne_half']].agg(['mean','count']).round(3))
print(ps[['insulin_x2','insulin_half']].describe().round(3))
print("corr ins x2 vs half per stay:", ps[['insulin_x2','insulin_half']].corr().iloc[0,1].round(3), " ne:", ps[['ne_x2','ne_half']].corr().iloc[0,1].round(3))
```