The 3% "omissions" were an artefact. They are all hour −1 readings, taken before the ICU stay, when no infusion can be given. From hour 0 on, insulin is always given when the reading is 150 or more. So no random omissions exist to test an all-or-nothing effect.

That leaves a gap. The within-tier noise shows that the exact dose size does not matter. But if the recorded noise were only bookkeeping and the patient really got the tier's standard dose, the tier steps themselves could still have an effect.

Next I will run proper regression-discontinuity estimates at 180, 220, 260 and 300. These use local linear fits on each side within ±20 mg/dL, excluding the exactly-at-threshold readings, which are mixed by rounding. Each outcome jump is divided by the dose jump. Per-u/h estimates near zero at all thresholds would settle insulin as inert in this ICU, for both the within-tier noise and the between-tier steps. Clear negative jumps would mean the recorded within-tier noise is not real dosing.

```python
def rd_est(D, c, h, ycol, dcol='ins'):
    S = D[(D.glucose_finger>=c-h)&(D.glucose_finger<=c+h)&(D.glucose_finger!=c)].copy()
    S = S[S[ycol].notna()]
    x = S.glucose_finger.values - c; a = (x>0).astype(float)
    X = np.column_stack([np.ones(len(S)), a, x, a*x])
    by,sey = ols(S[ycol].values, X, S.stay.values)
    bd,sed = ols(S[dcol].values, X, S.stay.values)
    return by[1], sey[1], bd[1], len(S)
for k in [1,2,3,4]:
    D[f'dR{k}'] = D[f'Rf{k}'] - D.glucose_finger
res = []
for c in [180,220,260,300]:
    for k in [1,2,3,4]:
        jy, se, jd, n = rd_est(D, c, 20, f'dR{k}')
        res.append((c,k,jy,se,jd,jy/jd,se/jd,n))
R_ = pd.DataFrame(res, columns=['thr','k','jump_y','se','jump_dose','per_unit','se_pu','n'])
print(R_.round(3).to_string())
# pooled per-unit by k (inverse-variance)
for k in [1,2,3,4]:
    r = R_[R_.k==k]; w = 1/r.se_pu**2
    print(k, "pooled per-unit", (r.per_unit*w).sum()/w.sum(), "+-", np.sqrt(1/w.sum()))
```