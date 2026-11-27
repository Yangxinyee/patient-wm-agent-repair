The insulin policy is a clean sliding scale on the glucose reading just taken (finger, or lab when there is no finger reading):

| glucose (mg/dL) | base rate (u/h) |
|---|---|
| 150–180 | 1 |
| 180–220 | 2 |
| 220–260 | 3 |
| 260–300 | 4 |
| 300+ | about 6 |

On top of the base rate there is multiplicative noise with log-sd about 0.31, rounded to 0.5 u/h, and the noise has mean zero. A reading of exactly 180 still gets base 1.

So within a band the dose varies randomly. If that noise is independent of the patient's hidden state, it identifies the insulin effect.

Next I'll compute the residual ε = log(rate) − log(base) and test whether it is exogenous:
- correlation with the previous residual (persistence within the stay);
- correlation with the past glucose change and with glucose_lab given the finger reading;
- correlation with MAP and norepinephrine.

All near zero would confirm independent noise. I'll also verify that lab-only hours follow the same scale.

```python
def base_ins(g):
    return np.select([g<150, g<=180, g<=220, g<=260, g<=300], [0,1,2,3,4], 6).astype(float)
# check thresholds precisely with integer readings near cutoffs
s = P[P.insulin.notna()].copy()
for cut in [150,180,220,260,300]:
    for gv in [cut-1, cut, cut+1]:
        x = s[s.g_now==gv].insulin
        print(cut, gv, len(x), round(np.log(x).mean(),3) if len(x) else None)
lab_only = s[s.glucose_finger.isna()]
lab_only = lab_only.assign(gb=(lab_only.glucose_lab//20)*20)
print("lab-only hours:", lab_only.groupby('gb').insulin.apply(lambda x: round(np.log(x).mean(),2)).to_dict())
```