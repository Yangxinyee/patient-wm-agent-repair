The original model hardly reacts to dose size. Its per-stay norepinephrine shifts have SD 0.07 mmHg around 0. Its per-stay insulin shifts (SD 0.34) correlate positively between ×2 and ×½ (r = 0.67), so path noise dominates rather than any dose response. Only 2 of the 60 norepinephrine stays start the infusion after hour 12, and the model does not exploit the start-dose confound there either. So the model answers about 0, and my estimates of the truth are about 0.

Because the "main readout" is insulin, I want to be sure I am not missing an insulin effect that is slow or switches with insulin on versus off. Two checks:

1. Regression-discontinuity estimates at the sliding-scale tiers for longer horizons (6 and 9 h), which hidden confounding cannot bias.
2. The on/off threshold at 150. Readings below 150 are only recorded about 30% of the time, so I will first check whether that measurement is random. If it is, outcomes measured on both sides are comparable.

A clear negative jump at longer horizons, or at 150, would show an insulin effect I have missed. Nulls would settle "no insulin effect" within about ±0.3 mg/dL per u/h.

```python
for k in [6,9]:
    D[f'dR{k}'] = D[f'Rf{k}'] - D.glucose_finger
res=[]
for c in [180,220,260,300]:
    for k in [6,9]:
        jy, se, jd, n = rd_est(D, c, 20, f'dR{k}')
        res.append((c,k,jy/jd,se/jd,n))
R6 = pd.DataFrame(res, columns=['thr','k','per_unit','se','n'])
for k in [6,9]:
    r = R6[R6.k==k]; w = 1/r.se**2
    print(f"k={k} tier-RD pooled per-unit {(r.per_unit*w).sum()/w.sum():+.3f} +- {np.sqrt(1/w.sum()):.3f}  ", r.per_unit.round(2).tolist())
# 150 threshold: measurement below 150 - is it random?
L = D[(D.glucose_finger<150)&(D.glucose_finger>=130)].copy()
L['meas_next'] = L.Rf1.notna().astype(float)
L['R_p1'] = P.glucose_finger.groupby(level=0).shift(1).reindex(L.index).values
print("P(next measured | R<150):", L.meas_next.mean().round(3))
print(L.groupby(L.glucose_finger//5*5).meas_next.mean().round(3).to_dict())
print("by hour:", L.groupby(L.hr//6*6).meas_next.mean().round(3).to_dict())
# run of hours since last measurement pattern: is there a schedule?
gaps = P.glucose_finger.notna().groupby(level=0).apply(lambda s: np.diff(np.where(s.values)[0]))
allg = np.concatenate(gaps.values)
print("gap distribution between finger readings:", pd.Series(allg).value_counts().sort_index().head(10).to_dict())
for k in [1,2,3]:
    jy, se, jd, n = rd_est(D, 150, 10, f'dR{k}')
    print(f"150-RD k={k}: jump {jy:+.2f}+-{se:.2f}, dose jump {jd:.2f}, per-unit {jy/jd:+.2f}+-{se/jd:.2f}, n={n}")
```