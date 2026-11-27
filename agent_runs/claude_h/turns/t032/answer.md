The norepinephrine start rule has a sharp edge. At an arterial MAP of 64 or lower it always starts (100%), at 65 it starts 59% of the time, and at 66 only 17%, falling off above. That is a strong discontinuity between 64 and 65 (integer readings).

Next I will run the fuzzy RD at a cutoff of 64.5. I will use a local linear fit on each side, bandwidth ±6 mmHg and stay-clustered SEs, for the jump in
- the MAP change over horizons 1–12;
- the cumulative norepinephrine dose received over the same horizon.

The ratio gives the effect per mcg/kg/min·hour. Near-zero MAP jumps despite a clear dose jump would confirm "absent" with variation independent of the titration noise.

```python
def rd_int(df, xcol, ycol, c, h):
    D = df[[xcol,ycol,'stay_id']].dropna()
    D = D[(D[xcol]>=c-h)&(D[xcol]<=c+h)]
    x = D[xcol]-c; bel = (x<0).astype(float)
    X = np.c_[np.ones(len(D)), bel, x, x*bel]
    b,se,_ = ols(D[ycol].values, X, D.stay_id.values)
    return b[1], se[1], len(D)
for k in [1,2,3,6,9,12]:
    bd, sed, n = rd_int(s0, 'ma', f'cum{k}', 64.5, 6)
    by, sey, n2 = rd_int(s0, 'ma', f'dM{k}', 64.5, 6)
    print(f"k={k}: cum-dose jump {bd:+.3f} ({sed:.3f}); dMAP jump {by:+.3f} ({sey:.3f}) n={n2}; per rate-hour {by/bd:+.3f}")
```