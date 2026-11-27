The insulin policy is a near-deterministic sliding scale on the reading just taken.

| Glucose (mg/dL) | Insulin |
|---|---|
| ≤ 149 | never given |
| 150 | given about half the time (a threshold at exactly 150) |
| 151–179 | ~1.05 U/h |
| 181–219 | ~2.1 U/h |
| 221–259 | ~3.15 U/h |
| ≥ 261 | ~4.2 U/h |

At the cut-points 180, 220 and 260 the rate is a 50/50 mix. Around each tier there is noise of ±0.5 U/h, and about 3–4% of indicated doses are skipped. Insulin is never given in an hour with no reading; the 7,942 cases without a finger-stick all have a lab glucose.

This points to the mechanism behind the model's error. In the rollout, readings from hour 13 on are hidden, but the recorded doses still encode the real hidden glucose through this tier rule. A doubled dose then looks like evidence of high glucose, so the model would read "×2" partly as "glucose was higher" rather than "more drug".

To estimate the true effect I need to know which reading drives the decision and whether the within-tier noise and the skips are exogenous. Next I will:
1. Test whether the finger or the lab value drives the tier when both exist.
2. Characterise the within-tier noise and the skips.
3. Check whether either correlates with pre-treatment variables: earlier glucose, the previous rate, MAP, norepinephrine, the hour. If they are exogenous, the noise is a usable natural experiment.

```python
def tier(g):
    return np.select([g<150, g<180, g<220, g<260], [0,1,2,3], 4).astype(float)
P = Ptr.copy()
P['g_prev'] = P.groupby(level=0).glucose_finger.shift(1)
P['ins_prev'] = P.groupby(level=0).insulin.shift(1)
both = P[P.glucose_finger.notna() & P.glucose_lab.notna()]
print("both:", len(both))
# which determines tier?
tb = both.copy()
tb['tf']=tier(tb.glucose_finger); tb['tl']=tier(tb.glucose_lab)
dis = tb[(tb.tf!=tb.tl) & (tb.glucose_finger.round()!=150) & (tb.glucose_lab.round()!=150)]
print("disagree:", len(dis))
print(dis.groupby(['tf','tl']).insulin.agg(['size','mean', lambda x: x.notna().mean()]).head(20))
# lab-only hours
lo = P[P.glucose_finger.isna() & P.glucose_lab.notna()].copy()
lo['tl']=tier(lo.glucose_lab)
print(lo.groupby('tl').insulin.agg(['size','mean', lambda x: x.notna().mean()]))
```