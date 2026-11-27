The discontinuity estimates agree with regression adjustment: insulin has no detectable effect on glucose. Pooled over the four thresholds, the per-u/h effect is:

| Horizon | Per-u/h effect (mg/dL) |
|---|---|
| 1 h | +0.03 ± 0.12 |
| 2 h | −0.12 ± 0.13 |
| 3 h | −0.17 ± 0.15 |
| 4 h | −0.02 ± 0.16 |

Both the between-tier steps and the within-tier noise therefore show that changing the insulin rate does nothing measurable to glucose. The true effect of insulin ×2 or ×½ should be close to 0, within a few tenths of a mg/dL. This matches the original model's +0.14 / −0.08. I will check this again later, but my working belief is that insulin is inert here.

Now norepinephrine → MAP. I will check the titration noise first: its residual autocorrelation within stay, which should be about 0 if it is random. Then I will estimate the effect of NE_t on the arterial MAP change 1–6 hours ahead. I control for the current arterial reading (fixed effects) and the previous rate, by demeaning and partialling out. A clear positive coefficient would mean norepinephrine really raises MAP. The original model's near-zero answers would then be badly wrong for norepinephrine.

```python
def fe_ols(y, X, g, cl):
    df = pd.DataFrame(np.column_stack([y, X]), columns=['y']+[f'x{i}' for i in range(X.shape[1])])
    df['g']=g; df['cl']=cl; df = df.dropna()
    cols = ['y']+[f'x{i}' for i in range(X.shape[1])]
    dm = df[cols] - df.groupby('g')[cols].transform('mean')
    b,se = ols(dm.y.values, dm[cols[1:]].values, df.cl.values)
    return b,se,len(df)
A = P[P.map_arterial.notna() & (P.index.get_level_values(1)>=0)].copy()
A['stay']=A.index.get_level_values(0); A['hr']=A.index.get_level_values(1)
maP = P.map_arterial.groupby(level=0)
for k in range(1,7):
    A[f'Mf{k}'] = maP.shift(-k).reindex(A.index).values
A['ne_p1'] = P.ne.groupby(level=0).shift(1).reindex(A.index).values
A['ma_p1'] = P.map_arterial.groupby(level=0).shift(1).reindex(A.index).values
# titration noise autocorrelation
T2 = A[(A.ne>0)&(A.ne_p1>0)&(A.ne<1)&(A.ne_p1<1)].copy()
T2['lr'] = np.log(T2.ne/T2.ne_p1)
T2['lr_res'] = T2.lr - T2.groupby(T2.map_arterial.round())['lr'].transform('mean')
T2['lr_res_p'] = T2.groupby('stay').lr_res.shift(1)
m = T2.lr_res_p.notna()
print("NE titration residual lag-1 corr:", np.corrcoef(T2.lr_res[m], T2.lr_res_p[m])[0,1].round(3), " n", m.sum())
g = A.map_arterial.round().astype(int).values
for k in range(1,7):
    y = (A[f'Mf{k}'] - A.map_arterial).values
    X = np.column_stack([A.ne.values, A.ne_p1.fillna(0).values])
    b,se,n = fe_ols(y, X, g, A.stay.values)
    print(f"k={k}: NE_t coef {b[0]:.3f} +- {se[0]:.3f} | NE_t-1 coef {b[1]:.3f} +- {se[1]:.3f}  n={n}")
```