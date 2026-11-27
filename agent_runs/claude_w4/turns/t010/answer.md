## Step 9: the insulin thresholds are sharp; the rate noise looks multiplicative

**What the last result showed.**
- The thresholds at finger glucose 150, 180, 220 and 260 are exact. At a recorded value of exactly 150 (or 180, 220, 260), the mean rate is halfway between the bands. That fits an unrounded reading being compared against the threshold.
- Inside a band the rate is spread around the band's base value (1, 2, 3, 4), and the spread grows with the base. That looks like multiplicative noise rounded to 0.5.
- Below 150 there is still about 6% "on", so something besides the last finger reading sometimes triggers insulin.

**What I will do next.** Two checks before the discontinuity estimates.
1. Explain the below-150 insulin: is the finger reading stale? Is a lab reading high? Is insulin continuing from the previous hour?
2. Check whether the within-band rate noise is persistent within a stay. A clinician-level factor would show high correlation of the band-normalised rate across hours of the same stay; per-record noise would show none. Also check whether it depends on observables.

**What would count as evidence.** If the noise is per-record and independent of everything observable, within-band variation is a strong source of identification alongside the threshold jumps.

```python
y = xi[(xi.band=='<=150')]
y_on = y[y.ins_rate>0]
print("below150 on: n", len(y_on), " age_finger dist", y_on.age_glucose_finger.value_counts().head(6).to_dict(), " vs all below150:", y.age_glucose_finger.value_counts(normalize=True).head(6).round(3).to_dict())
print("prev ins on frac among below150-on:", (y_on.prev_ins_rate>0).mean(), " among below150-off:", (y[y.ins_rate==0].prev_ins_rate>0).mean())
print("last lab among below150 on:", y_on.last_glucose_lab.describe().round(1).to_dict())
# Is the threshold applied to the reading at current hour only? Look at on-rate by age when last finger in (150,180]
z = xi[xi.band=='150-180']
print("150-180: on by age:", z.groupby('age_glucose_finger').ins_rate.apply(lambda s:(s>0).mean()).head(8).round(3).to_dict())
y['fresh'] = y.age_glucose_finger==0
print(y.groupby(['fresh', y.prev_ins_rate>0]).ins_rate.apply(lambda s:(s>0).mean()).round(3))
# within-band normalized rate persistence
base = {'150-180':1.0,'180-220':2.0,'220-260':3.0,'>260':4.0}
w = xi[xi.band.isin(base.keys()) & (xi.ins_rate>0)].copy()
w['ratio'] = w.ins_rate / w.band.map(base).astype(float)
w['lr'] = np.log(w.ratio)
print("log ratio by band:", w.groupby('band').lr.agg(['mean','std']).round(3).to_dict())
w = w.sort_values(['stay_id','h'])
w['lr_prev'] = w.groupby('stay_id').lr.shift(1)
w['h_prev'] = w.groupby('stay_id').h.shift(1)
ww = w[(w.h - w.h_prev)==1]
print("corr lr with prev-hour lr (same stay):", np.corrcoef(ww.lr, ww.lr_prev)[0,1], len(ww))
sm = w.groupby('stay_id').lr.agg(['mean','count'])
print("stay-mean lr var:", sm[sm['count']>=5]['mean'].var(), " pooled var:", w.lr.var())
```