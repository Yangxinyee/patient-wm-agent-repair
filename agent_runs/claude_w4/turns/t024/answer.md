## Step 23: norepinephrine is titrated as a multiplicative random walk with MAP thresholds

**What the last result showed.** After hour 0, norepinephrine starts exactly when the latest MAP is below 65, with no exceptions either way. While running, the log rate-change depends on the MAP band:

| latest MAP | drift in log rate | stop probability |
|---|---|---|
| below 65 | about +0.05 to +0.16 (lower at very low MAP, where many rates already sit at the 1.0 cap) | 0 |
| 65 to 75 | about −0.02 (roughly hold) | small (0.7–4%) |
| above 75 | about −0.23 | about 19% |

The noise around the drift has SD about 0.29–0.35 in every band. So the rate follows a random walk with i.i.d.-looking steps, clipped to [0.01, 1.0].

**What this means.** As with insulin, the random step at each hour is a candidate instrument. Unlike insulin, each surprise persists in all later rates, because each rate builds on the last. The two-stage least squares design handles that, since the first stage captures the persistence.

**What I will do next.**
1. Check that the norepinephrine step noise has no autocorrelation.
2. Check whether MAP measurement (cuff and arterial) depends on the patient's state, to choose an outcome free of selection.

**What would count as evidence.** A random, state-independent measurement schedule for some MAP reading, like lab glucose, would give a clean outcome. Otherwise I will need care with selection.

```python
on2 = on[on.ne_rate>0].copy()
on2['band3'] = np.where(on2.lm_art_pref<65,0,np.where(on2.lm_art_pref<=75,1,2))
on2['res'] = on2.lr - on2.groupby('band3').lr.transform('mean')
on2 = on2.sort_values(['stay_id','h'])
on2['res_prev'] = on2.groupby('stay_id').res.shift(1); on2['hp'] = on2.groupby('stay_id').h.shift(1)
q = on2[(on2.h-on2.hp==1) & (on2.ne_prev<1.0) & (on2.ne_rate<1.0)]
print("NE step-noise autocorr:", np.corrcoef(q.res, q.res_prev)[0,1], len(q))
xx = xn.copy()
xx['has_c'] = xx.map_cuff.notna(); xx['has_a'] = xx.map_arterial.notna()
xx['lag_m'] = xx.groupby('stay_id').lm_art_pref.shift(1)
xx['lag_ne'] = xx.prev_ne_rate>0
art_stay = xx.groupby('stay_id').has_a.transform('any')
xx['art_stay'] = art_stay
print("P(cuff) by prev MAP bin, art stays vs not:")
print(xx.groupby(['art_stay', pd.cut(xx.lag_m,[0,55,65,75,85,200])]).has_c.mean().round(3).to_string())
print("P(art | art stay) by prev MAP:", xx[xx.art_stay].groupby(pd.cut(xx.lag_m,[0,55,65,75,85,200])).has_a.mean().round(3).to_dict())
print("P(cuff) by lag NE on:", xx.groupby(['art_stay','lag_ne']).has_c.mean().round(3).to_dict())
# arterial line placement: first hour of arterial
fa = xx[xx.has_a].groupby('stay_id').h.min()
print("first art hour dist:", fa.value_counts().sort_index().head(10).to_dict())
```