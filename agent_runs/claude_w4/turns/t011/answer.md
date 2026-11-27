## Step 10: insulin rate noise is independent per record; the scale uses the latest glucose of either kind

**What the last result showed.**
- When insulin runs although the last finger reading is at or below 150, the last lab glucose is almost always above 150. So the sliding scale is applied to the most recent glucose reading, finger or lab.
- Within a band, the multiplicative rate noise (log SD about 0.31–0.37) has zero correlation with the previous hour's noise in the same stay. Its stay-level variance (0.0087) is what pure per-record noise predicts. So it is i.i.d. per record.
- In the 150–180 band the finger reading is never more than one hour old. Glucose monitoring becomes hourly once glucose is high, so how often readings are taken depends on the patient's state.

**What this means.** If that i.i.d. noise is also independent of the hidden patient state, within-band dose variation is as good as random. That would give a high-power estimate of the insulin effect, which I can cross-check against the discontinuity jumps at the thresholds.

**What I will do next.**
1. Pin down the scale's input exactly: which reading counts as "latest" when finger and lab fall in the same hour.
2. Test whether the noise relates to observables: the current reading within its band, the previous glucose, MAP, norepinephrine, the hour.

**What would count as evidence.** If the noise is unrelated to all of these, I can treat it as random. If it is related, that suggests the clinician sees something more, and the noise may be confounded.

```python
x = g[g.h.between(0,24)].copy()
# latest glucose: if both at same hour?
both = x[x.glucose_finger.notna() & x.glucose_lab.notna()]
def band_of(v):
    return np.digitize(v, [150,180,220,260], right=True)  # 0: <=150, 1:(150,180], ...
def rate_band(r):
    return np.where(r==0,0,np.where(r<1.5,1,np.where(r<2.5,2,np.where(r<3.5,3,4))))
bf = band_of(both.glucose_finger.values); bl = band_of(both.glucose_lab.values); rb = rate_band(both.ins_rate.values)
dis = bf!=bl
print("same-hour both: n", len(both), " band disagree:", dis.sum(), " on-off agree with finger:", ((bf>0)==(both.ins_rate.values>0))[dis].mean(), " with lab:", ((bl>0)==(both.ins_rate.values>0))[dis].mean())
print("   mean rate by (bf,bl) where disagree:", pd.DataFrame({'bf':bf[dis],'bl':bl[dis],'r':both.ins_rate.values[dis]}).groupby(['bf','bl']).r.agg(['mean','size']).head(12).to_dict())
# define latest glucose, prefer lab when same hour? test both versions
def latest(x, prefer):
    af = x.age_glucose_finger.fillna(99); al = x.age_glucose_lab.fillna(99)
    lf = x.last_glucose_finger; ll = x.last_glucose_lab
    if prefer=='lab': use_lab = al<=af
    else: use_lab = al<af
    return np.where(use_lab, ll, lf)
for pref in ['lab','finger']:
    lg = latest(x, pref)
    ok = ~np.isnan(lg)
    b = band_of(lg[ok]); r = x.ins_rate.values[ok]
    print(pref, "on-off agreement:", ((b>0)==(r>0)).mean(), " band agreement among on:", (b==rate_band(r))[r>0].mean())
```