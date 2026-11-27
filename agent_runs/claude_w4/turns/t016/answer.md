## Step 15: check whether lab glucose is measured on a schedule

**What the last result showed.** The dose surprise (rate minus its band mean) is well defined: mean about 0, SD 0.61 units/h. Expected rates by band are 1.06, 2.10, 3.15 and 4.28.

**A problem for multi-hour effects.** Finger readings become hourly once glucose is above 150. A dose surprise can therefore change whether a later finger reading exists at all: more insulin means lower glucose, which means fewer checks. Regressing on finger outcomes at longer horizons would then suffer from selection after treatment.

**What I will do next.** Check whether lab glucose (and, for norepinephrine later, cuff MAP) follows a fixed schedule that does not depend on the patient's state:
- which hours labs fall on;
- whether lab presence depends on the previous glucose or on insulin.

**What would count as evidence.** A fixed schedule (for example every 6 hours) would make lab glucose a selection-free outcome for the impulse response. If lab timing depends on state, I will need a different strategy, such as conditioning on stays that stay well above 150, or a state-space model.

```python
xx = x.copy()
xx['has_lab'] = xx.glucose_lab.notna(); xx['has_f'] = xx.glucose_finger.notna()
print("lab presence by hour:", xx.groupby('h').has_lab.mean().round(3).to_dict())
# interval between labs within stay
labh = xx[xx.has_lab][['stay_id','h']]
labh['dh'] = labh.groupby('stay_id').h.diff()
print("lab interval dist:", labh.dh.value_counts().sort_index().head(15).to_dict())
xx['lag_g'] = xx.groupby('stay_id').lg.shift(1)
xx['gbin'] = pd.cut(xx.lag_g,[0,100,130,150,180,220,400])
print("P(lab) by prev latest glucose:", xx.groupby('gbin').has_lab.mean().round(3).to_dict())
print("P(finger) by prev latest glucose:", xx.groupby('gbin').has_f.mean().round(3).to_dict())
print("P(lab) by prev insulin on:", xx.groupby(xx.prev_ins_rate>0).has_lab.mean().round(3).to_dict())
# number of labs per stay
print("labs per stay:", xx.groupby('stay_id').has_lab.sum().value_counts().sort_index().to_dict())
```