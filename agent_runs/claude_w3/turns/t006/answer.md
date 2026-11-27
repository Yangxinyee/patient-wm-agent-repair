The dosing is protocol-like and driven by the latest reading. Insulin is always on when the last finger glucose is above 150 and essentially never on below 120. Its rate steps up with glucose, to about 1 unit/h at 150–180, 2 at 180–220, 3 at 220–250 and about 4 above 250, with ±0.5 jitter. Norepinephrine is always on when MAP is below 60, partly on at 60–80, and its rate rises as MAP falls. The few insulin starts at 120–150 suggest the policy may also use the lab glucose or something else. Stays that ever receive insulin have much higher glucose (169 vs 110), and stays on norepinephrine have lower MAP.

This means the records come with two possible sources of identification: sharp thresholds (a regression-discontinuity design) and jitter of the rate within bands. Both are valid only if the policy depends on recorded quantities. So next I will pin the policy down exactly:
- On/off and rate in 1 mg/dL steps around 150, 180, 220 and 250, using only hours where the finger reading is fresh (age 0).
- Whether the lab glucose matters.
- Whether the within-band jitter is predictable from anything recorded, such as the previous rate or a simultaneous lab reading.

If the jitter is predicted by a simultaneous lab reading given the finger reading, dosing depends on more than the finger value, and I would have to treat that as a possible hidden confounder.

```python
q = pn[(pn.h>=0)&(pn.glucose_finger_age==0)].copy()
q['gf'] = q.glucose_finger
# fine-grained on/off near 150
near = q[(q.gf>=140)&(q.gf<=160)].groupby('gf').agg(n=('ins_on','size'), p_on=('ins_on','mean'), r=('insulin','mean'))
print(near.round(3).T.to_string())
# rate near 180, 220, 250
for lo,hi in [(170,190),(210,230),(240,260)]:
    nr = q[(q.gf>=lo)&(q.gf<=hi)].groupby('gf').agg(n=('insulin','size'), r=('insulin','mean'), sd=('insulin','std'))
    print(nr.round(2).T.to_string())
# distribution of rate within bands
for lo,hi in [(151,180),(181,220),(221,250),(251,400)]:
    sub = q[(q.gf>=lo)&(q.gf<=hi)]
    print((lo,hi), sub.insulin.value_counts(normalize=True).sort_index().round(3).to_dict())
# insulin on at low finger glucose: what explains it?
low = pn[(pn.h>=0)&(pn.glucose_finger_last<=150)&(pn.ins_on==1)]
print("n on with last finger<=150:", len(low))
print(low[['glucose_finger_last','glucose_finger_age','glucose_lab_last','glucose_lab_age','ins_prev','insulin']].describe().round(2).to_string())
```