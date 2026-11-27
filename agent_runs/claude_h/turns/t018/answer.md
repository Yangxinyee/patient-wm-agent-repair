The norepinephrine titration has an iid innovation. Given the MAP bin and previous-rate bin, the residual SD is 0.138, its autocorrelation is −0.001, and it is unrelated to the MAP trend and the hour. A shock persists in later rates (0.70 → 0.17 over 8 hours), as titration implies.

The MAP response to the shock is negligible: +0.14 ± 0.09 mmHg per mcg/kg/min at 1 hour, and −0.05 to −0.34 (SE 0.10–0.18) at 2–8 hours. A full 1 mcg/kg/min change, which spans the entire dose range, would move MAP by less than about 0.4 mmHg. So norepinephrine, like insulin, appears to have essentially no effect in this ICU.

Two null effects need a robustness check before I build on them. Next I will run a regression-discontinuity test for insulin at the 180 and 220 scale thresholds. There the mean dose doubles (1 → 2, 2 → 3) between readings a few mg/dL apart, and the next reading is always observed because insulin is on. If insulin truly has no effect, the change G(t+k) − G(t) should show no jump at the threshold beyond the smooth trend.

```python
def rd(df, xcol, ycol, c, h, extra_mask=None):
    D = df[[xcol,ycol,'stay_id']].dropna()
    D = D[(D[xcol]>=c-h)&(D[xcol]<c+h)]
    x = D[xcol]-c; above = (x>=0).astype(float)
    X = np.c_[np.ones(len(D)), above, x, x*above]
    b,se,_ = ols(D[ycol].values, X, D.stay_id.values)
    return b[1], se[1], len(D)
d = P[P.gf.notna()].copy().reset_index()
for k in [1,2,3,4,6]:
    d[f'dG{k}'] = d[f'gf_f{k}'] - d.gf
for c in [180, 220, 260]:
    dd = d[d.ins>0]
    jump_dose = rd(dd, 'gf', 'ins', c, 20)
    print(f"threshold {c}: dose jump {jump_dose[0]:+.3f} ({jump_dose[1]:.3f})")
    for k in [1,2,3,4,6]:
        b,se,n = rd(dd, 'gf', f'dG{k}', c, 20)
        print(f"   k={k}: jump in dG {b:+.2f} ({se:.2f}) n={n}  -> per unit dose {b/jump_dose[0]:+.2f}")
```