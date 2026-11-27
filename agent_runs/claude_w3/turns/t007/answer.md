The insulin policy is now clear and memoryless:
- Insulin starts exactly when the most recent glucose exceeds 150. At a recorded 150, 52% of decisions are on, which suggests the threshold acts on an unrounded value.
- The base rate is 1, 2 or 3 units/h with breaks at 180 and 220, and higher above 250.
- On top of the base rate there is a multiplicative jitter: the spread grows with the band, 0.36, 0.66, then about 0.97.
- The rare "on" cases with an old finger reading of 150 or below are explained by a more recent lab glucose (about 158, usually at age 0). So the policy reads whichever glucose is most recent, finger or lab.

The jitter within a band is the natural source of variation, but only if it is truly random. Next I will:
- Define a "policy glucose" (the most recent reading, checking which one wins when finger and lab share an hour) and the base rate.
- Compute jitter = rate / base.
- Test whether the jitter is persistent within a stay, related to the trend, related to the stay's overall glucose level, or related to a simultaneous lab-minus-finger difference.

If the jitter is i.i.d. and unrelated to all of these, the records give clean leverage on the insulin effect. A stay-level component would be a warning sign of confounding by patient traits.

```python
# Which reading does the policy use when finger & lab are at the same hour?
both_now = q[(q.glucose_lab_age==0)]
def onrule(x): return (x>150).astype(int)
for nm, val in [('finger', both_now.gf), ('lab', both_now.glucose_lab), ('mean', (both_now.gf+both_now.glucose_lab)/2),
                ('max', np.maximum(both_now.gf,both_now.glucose_lab))]:
    print(nm, "agreement of on/off:", (onrule(val)==both_now.ins_on).mean().round(4))
# general: latest reading, lab-only hours
lab_only = pn[(pn.h>=0)&(pn.glucose_lab_age==0)&(pn.glucose_finger_age!=0)]
print("lab-only hours agreement with lab>150:", ((lab_only.glucose_lab>150).astype(int)==lab_only.ins_on).mean().round(4), len(lab_only))
# no fresh glucose at all: is policy based on latest (aged) reading?
none = pn[(pn.h>=0)&(pn.glucose_lab_age!=0)&(pn.glucose_finger_age!=0)].copy()
lf = none.glucose_finger_age.fillna(99); ll = none.glucose_lab_age.fillna(99)
none['pol'] = np.where(lf<=ll, none.glucose_finger_last, none.glucose_lab_last)
print("no-fresh hours: agreement with latest>150:", ((none.pol>150).astype(int)==none.ins_on).mean().round(4), len(none))
print("no-fresh hours, p_on:", none.ins_on.mean().round(3))
```