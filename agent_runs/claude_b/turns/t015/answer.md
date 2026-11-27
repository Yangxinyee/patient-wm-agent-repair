Each threshold shows two things.

- **Rounding at the threshold.** At readings exactly on a threshold (150, 180, 220, 260, 300) the mean dose sits halfway between tiers. The readings are probably rounded from a continuous value that the policy uses.
- **Next reading depends on insulin.** Below 150 the next hour's finger reading is recorded only 30% of the time; above 150 it is 97%. So the 150 threshold is unusable for a discontinuity comparison. At the other thresholds about 95% are observed on both sides.

Eyeballing the jump in next-hour change (dose jump in brackets):

| Threshold | Jump in R_{t+1} − R_t |
|---|---|
| 180 (+1 u/h) | about −0.5 |
| 220 (+1 u/h) | about −0.3 |
| 260 (+1 u/h) | about −0.2 |
| 300 (+2 u/h) | about −0.5 |

So the one-hour effect of insulin looks small, perhaps −0.3 mg/dL per u/h. But the effect could be delayed or build up over hours, which matters a lot for a 12-hour readout.

Before fitting dynamics I need the measurement structure: how finger and lab glucose relate, and arterial and cuff MAP. Measurement noise drives apparent mean reversion and determines how well a reading pins down the true state. I will compare paired readings taken in the same hour and look at the hour-to-hour autocorrelations.

```python
both_g = P[P.glucose_finger.notna() & P.glucose_lab.notna()]
d = both_g.glucose_finger - both_g.glucose_lab
print("finger-lab: n", len(d), "mean", d.mean().round(2), "sd", d.std().round(2), " ratio mean", (both_g.glucose_finger/both_g.glucose_lab).mean().round(3))
# fit finger = a + b*lab
A = np.vstack([np.ones(len(both_g)), both_g.glucose_lab]).T
print("finger ~ lab:", np.linalg.lstsq(A, both_g.glucose_finger.values, rcond=None)[0].round(3))
A = np.vstack([np.ones(len(both_g)), both_g.glucose_finger]).T
print("lab ~ finger:", np.linalg.lstsq(A, both_g.glucose_lab.values, rcond=None)[0].round(3))
both_m = P[P.map_arterial.notna() & P.map_cuff.notna()]
d2 = both_m.map_arterial - both_m.map_cuff
print("art-cuff: n", len(d2), "mean", d2.mean().round(2), "sd", d2.std().round(2))
A = np.vstack([np.ones(len(both_m)), both_m.map_cuff]).T
print("art ~ cuff:", np.linalg.lstsq(A, both_m.map_arterial.values, rcond=None)[0].round(3))
A = np.vstack([np.ones(len(both_m)), both_m.map_arterial]).T
print("cuff ~ art:", np.linalg.lstsq(A, both_m.map_cuff.values, rcond=None)[0].round(3))
# consecutive-hour differences
for c in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
    x = P[c]; y = P[f'{c}_f1']; m = x.notna()&y.notna()
    print(c, "lag1 diff sd", (y[m]-x[m]).std().round(2), " corr", np.corrcoef(x[m],y[m])[0,1].round(4), " sd level", x.std().round(1))
    y2 = P[f'{c}_f2']; m2 = x.notna()&y2.notna()
    print("   lag2 diff sd", (y2[m2]-x[m2]).std().round(2), " corr", np.corrcoef(x[m2],y2[m2])[0,1].round(4))
```