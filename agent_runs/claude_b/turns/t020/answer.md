No horizon from 1 to 9 hours shows an insulin effect. Every estimate lies within about ±0.1–0.15 mg/dL per u/h of zero, and the largest ratio to its standard error is about 1.6. So varying the insulin rate within a tier does nothing to glucose, at once or later.

That leaves one loophole. The effect might be all-or-nothing or saturating: any insulin lowers glucose, but more insulin adds nothing. Within-tier noise and the step-ups at 180 and above cannot see such an effect, and a ×2 or ×½ edit might then do almost nothing either.

The about 3% of doses omitted at readings above 150 are a natural test. Next I will compare the subsequent glucose changes after an omitted dose with those after a given dose at the same reading, and check whether omissions look random. A clear difference would mean an on/off effect. No difference would mean insulin is inert in this ICU.

```python
D2 = D[(D.glucose_finger>=155)&(D.glucose_finger<300)].copy()
D2['given'] = (D2.ins>0).astype(float)
print("omission rate", 1-D2.given.mean())
# omissions vs history: previous dose given? previous reading?
D2['ins_p1'] = P.ins.groupby(level=0).shift(1).reindex(D2.index).values
D2['R_p1'] = P.glucose_finger.groupby(level=0).shift(1).reindex(D2.index).values
print(D2.groupby('given')[['glucose_finger','ins_p1','R_p1','hr']].mean().round(2))
g = D2.glucose_finger.round().astype(int).values
for k in range(1,7):
    y = (D2[f'Rf{k}']-D2.glucose_finger).values
    b,se,n = fe_slope(y, D2.given.values, g, D2.stay.values)
    obs = D2[f'Rf{k}'].notna().groupby(D2.given).mean().round(3).to_dict()
    print(f"k={k}: given vs omitted on R_t+{k}: {b:.2f} +- {se:.2f}  n={n} obs-rate {obs}")
```