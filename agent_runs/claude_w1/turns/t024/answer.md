The deconvolution changes the insulin picture. Clinician feedback is small per hour (later doses fall by about 0.06 u/h), but it adds up. Holding later doses fixed, the response to a one-hour pulse has a long tail: −2.08, −2.41, −2.16, −1.79, −1.26, −0.95 … still −0.30 at 12 hours. The cumulative effect is −13.7 mg/dL per u/h·h by 12 hours and still growing.

The norepinephrine response is shorter-lived: +3.4, +2.6, +1.6, +0.9, +0.4 mmHg …, about 0 from 7 hours on. Its cumulative effect is about +9 mmHg per mcg/kg/min·h.

Cuff MAP reads about 3 mmHg above arterial, with sd 6.7.

The insulin tail rests on small numbers, each with SE ~0.1, so it needs checking. The noise-based estimates also assume the dose noise is unrelated to anything hidden. A regression-discontinuity check at the 180 and 220 thresholds tests that independently. If the jump in next-hour glucose there matches about −2.1 mg/dL per u/h, the noise-based identification holds. If it differs, there's hidden confounding. First I'll save these findings to the notebook.

```python
notes = """
## Runs 11-23: causal effect estimates (train.csv), using exogenous dose noise
- Insulin noise log(rate/base) iid: corr with prev noise 0.000, with G/MAP/NE/h ~0.01. NE innovation (log ratio resid) corr prev -0.014, MAP -0.04.
- Measurement: finger vs lab same hour: no bias, sd diff ~5.5% of G (multiplicative). cuff - arterial = +3.0 mmHg, sd 6.7.
- No cross effects: NE->glucose ~0, insulin->MAP ~0.
- Insulin -> glucose: local projection total effect per +1 u/h shock: k=1..6: -2.08,-2.29,-1.87,-1.37,-0.76,-0.45 (->0 by 12).
  Linear in dose (shock^2 ~0). Mildly larger at high G: -1.83 (G~164), -2.07 (196), -2.21 (234), -2.43 (272) at k=1.
  Feedback on later doses small: -0.06 u/h. Deconvolved fixed-dose impulse response h_ins(k):
  -2.08,-2.41,-2.16,-1.79,-1.26,-0.95,-0.77,-0.63,-0.55,-0.43,-0.33,-0.30 (cum -13.7 at 12h)  [tail uncertain]
- NE -> MAP_arterial: total per +1 mcg/kg/min innovation: 3.4,5.1,5.4,5.1,4.4,3.8,...; dose persistence 0.73,0.56,0.44,...
  Deconvolved h_ne(k): 3.40,2.60,1.62,0.94,0.41,0.33,0.0,... (cum ~9.3). Linear in rate (k=1 effect 3.3-3.6 across rate levels).
- Model-side caution: NE rate>1.0 / dose>4.8 -> bins 13-16 never seen in training (x2 edit hits them).
"""
with open('/work/notes.md','a') as f: f.write(notes)

# RD at 180 and 220 for insulin (finger reading at hour s, insulin on)
def rd(df, thr, bw, k, outcol='G'):
    sub = df[(df.glucose_finger>=thr-bw)&(df.glucose_finger<thr+bw)&df[f'{outcol}+{k}'].notna()].copy()
    x = sub.glucose_finger - thr
    above = (x>=0.5).astype(float)  # 180 exactly is mixed; drop exact threshold
    keep = (x!=0)
    sub=sub[keep]; x=x[keep]; above=above[keep]
    X = np.column_stack([np.ones(len(sub)), x, above, above*x])
    r,_ = ols(sub[f'{outcol}+{k}'], X, ['c','x','jump','jx'], cluster=sub.stay_id)
    Xi = X.copy()
    ri,_ = ols(sub.insulin, X, ['c','x','jump','jx'], cluster=sub.stay_id)
    return r.loc['jump','b'], r.loc['jump','se'], ri.loc['jump','b'], len(sub)
Sf = S[S.glucose_finger.notna()]
for thr in [180,220,260]:
    for k in [1,2,3,4]:
        j, se, ji, n = rd(Sf, thr, 15, k)
        print(thr, k, n, "jump G", round(j,3), "se", round(se,3), "jump I", round(ji,3), "ratio", round(j/ji,3))
```