On hours with insulin > 0, the one-hour effect of the jittered insulin dose on the next finger glucose is essentially zero: +0.001 ± 0.043 mg/dL per u/h with glucose-bin fixed effects, and −0.015 ± 0.039 with trend and hour controls. It is stable across specifications. The earlier −0.39 came from the zero-dose rows (selection and omission). So insulin does not move glucose within the first hour.

The effect could still be delayed, or act on the latent level and show up later. Next I will run local projections: G(t+k) − G(t) on the jittered dose at t for k = 1..8, with fixed effects absorbed by demeaning (which fixes the singular-matrix problem). Because the jitter is iid, these give the closed-loop total effect at each horizon. All-zero horizons would mean insulin has no effect here. A growing negative response would mean a delayed effect, which I would then convert to the open-loop answer.

```python
def absorb(df, cols, groups):
    # demean cols within groups (single FE)
    gm = df.groupby(groups)[cols].transform('mean')
    return df[cols] - gm
def fe_ols(df, ycol, xcols, fecol, clcol='stay_id'):
    D = df[[ycol]+xcols+[fecol, clcol]].dropna()
    A = absorb(D, [ycol]+xcols, D[fecol])
    b, se, r = ols(A[ycol].values, A[xcols].values, D[clcol].values)
    return b, se, len(D)
P['gbin5'] = (P.gf.clip(40,450)//5)
for k in range(1,9):
    P[f'gf_f{k}'] = g.gf.shift(-k)
    P[f'ins_f{k}'] = g.ins.shift(-k)
d = P[(P.ins>0) & P.gf.notna()].copy().reset_index()
for k in range(1,9):
    d[f'dG{k}'] = d[f'gf_f{k}'] - d.gf
    b,se,n = fe_ols(d, f'dG{k}', ['ins'], 'gbin5')
    # observation rate of the outcome
    obs = d[f'gf_f{k}'].notna().mean()
    # dose feedback: effect of ins_t on ins_{t+k}
    b2,se2,_ = fe_ols(d, f'ins_f{k}', ['ins'], 'gbin5')
    print(f"k={k}: dG coef {b[0]:+.3f} ({se[0]:.3f}) n={n} obs={obs:.3f} | future dose coef {b2[0]:+.3f} ({se2[0]:.3f})")
```