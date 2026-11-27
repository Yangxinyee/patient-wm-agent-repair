A dose deviation at hour t lowers glucose_finger by 2.0 at t+1 and 2.25 at t+2. The effect then decays: -1.8, -1.3, -0.7, -0.45 at lags 3–6. So insulin's effect persists, peaks around 2 hours and fades over about 5–6 hours. Doubling insulin from hour 12 should therefore build up a cumulative shift of several mg/dL per unit/h, not a one-hour blip.

These are total effects, though, including the clinicians' feedback: a lower glucose leads to lower later doses. The evaluation holds later doses at their recorded values times the multiplier. Next I'll measure that feedback, i.e. how a dose deviation at t changes the doses at t+1..t+6, and whether it changes the chance of a finger reading being taken. Near-zero feedback would let me read these lag coefficients directly as the fixed-dose impulse response. Sizeable feedback would mean I must deconvolve it, likely making the fixed-dose effect larger and longer-lasting than these numbers.

```python
rows=[]
for t in range(0,24):
    j=col(t)
    b = band_of(glast[:,j])
    d = {'t':t,'g0':gf[:,j],'band':b,'ins':ins[:,j]}
    for k in range(1,7):
        jj = j+k
        d[f'i{k}'] = ins[:,jj] if jj <= col(24) else np.nan
        d[f'p{k}'] = np.isfinite(gf[:,jj]).astype(float) if jj <= col(24) else np.nan
    rows.append(pd.DataFrame(d))
F = pd.concat(rows, ignore_index=True)
F = F[F.ins>0].copy()
F['dev'] = F.ins - F.groupby('band').ins.transform('mean')
for k in range(1,7):
    s = F.dropna(subset=[f'i{k}'])
    X = [s.dev.values] + [(s.band==b).values.astype(float) for b in sorted(s.band.unique())[1:]]
    r1 = ols(s[f'i{k}'].values, X, ['dev']+[f'b{b}' for b in sorted(s.band.unique())[1:]])
    r2 = ols(s[f'p{k}'].values, X, ['dev']+[f'b{b}' for b in sorted(s.band.unique())[1:]])
    print(k, "dose feedback", r1.loc['dev','b'].round(4), "+-", r1.loc['dev','se'].round(4), " finger-presence", r2.loc['dev','b'].round(4), "+-", r2.loc['dev','se'].round(4))
```