`Dtr2` is the one-step training design: next finger reading `y`, current reading `g0`, tier, earlier readings, lab values, dose deviations `dev1`–`dev6` and raw doses.

To test linearity, I regress `y` on:
- tier-specific quadratics in `g0`;
- earlier readings and lab values, with missing-value flags;
- `dev2`–`dev6`;
- `dev1` split into its positive and negative parts.

If the effect is linear, the two parts have equal slopes (about −2). In the records, positive deviations reach about +1× the base dose and negative ones about −0.5×. So a curved true effect would show up as unequal slopes, and the true ×2/×½ ratio would then not be 2.

As a second check, I fit `dev1` with a squared term.

```python
Dd_ = Dtr2.dropna(subset=['y','g0']).copy()
t_ = Dd_.tier.astype(int).values; g0c = (Dd_.g0.values - 180)/50
cols_ = []
for k in range(6):
    mk = (t_==k).astype(float); cols_ += [mk, mk*g0c, mk*g0c**2]
for c in ['g1','g2','g3','lab','lab1']:
    v = Dd_[c].values.astype(float); miss = np.isnan(v)
    cols_ += [np.where(miss, 0, v - Dd_.g0.values), miss.astype(float)]
for j in range(2, 7): cols_.append(np.nan_to_num(Dd_[f'dev{j}'].values.astype(float)))
d1_ = np.nan_to_num(Dd_.dev1.values.astype(float))
def ols_se(Xm, yv):
    b = np.linalg.lstsq(Xm, yv, rcond=None)[0]; r = yv - Xm @ b
    XtXi = np.linalg.pinv(Xm.T @ Xm); s2 = r.var()
    return b, np.sqrt(np.diag(XtXi) * s2)
yv_ = Dd_.y.values.astype(float)
Xa = np.column_stack(cols_ + [np.maximum(d1_,0), np.minimum(d1_,0)])
b_, se_ = ols_se(Xa, yv_)
print('n', len(yv_), '| dev1 pos slope %.3f (se %.3f)  neg slope %.3f (se %.3f)' % (b_[-2], se_[-2], b_[-1], se_[-1]))
Xb = np.column_stack(cols_ + [d1_, d1_**2])
b2_, se2_ = ols_se(Xb, yv_)
print('dev1 lin %.3f (se %.3f)  dev1^2 %.3f (se %.3f)' % (b2_[-2], se2_[-2], b2_[-1], se2_[-1]))
print('dev1 range: pct1/99', np.percentile(d1_[d1_!=0], [1, 99]).round(2), 'frac>0 %.2f' % (d1_>0).mean())
```