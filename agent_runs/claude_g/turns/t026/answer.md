With richer pre-dose controls, the insulin coefficients collapse to 0: +0.013 ± 0.040 mg/dL per u/h at lag 1, and within about 1 SE of 0 at all lags (h ≥ 12 included). The multiplier is unrelated to the previous reading given the current one (partial correlation 0.003). The earlier positive drift was imperfect adjustment, not an effect. Insulin's true effect on glucose_finger is 0 by three independent designs: random margin, step-ups at thresholds, and start at 150.

For norepinephrine the DML estimate was a small −0.33 ± 0.08 mmHg per unit, and the start RD was null. I want an estimate as clean as insulin's, plus subgroup checks, before deciding whether the original model (which already reads about 0) needs any change. Next:
1. For continuing infusions below the 1.0 cap, define the random titration residual: log(rate ÷ previous rate) minus its mean given MAP at h. Check that it is iid and unrelated to past MAP. Then regress later MAP change on the implied dose deviation, with history controls.
2. Repeat the insulin margin estimate in subgroups (on or off norepinephrine; arterial line or not), to rule out an effect hidden in an interaction.

```python
# NE: continuing infusions, not capped previous & current
H3 = H.copy()
H3['rate'] = H3['ne']; H3['lr'] = np.log(H3['rate']/H3['ne_m1'])
c = H3[(H3['rate']>0) & (H3['ne_m1']>0) & H3['map_arterial'].notna() & (H3['rate']<1.0) & (H3['ne_m1']<1.0)].copy()
c['mr'] = c['map_arterial'].round()
c['lr_res'] = c['lr'] - c.groupby('mr')['lr'].transform('mean')
c['dev'] = c['rate'] - c['ne_m1']*np.exp(c.groupby('mr')['lr'].transform('mean'))
# independence checks
for col in ['ma_m1','ma_m2','ne_m2','hh','mc']:
    s = c[[col,'lr_res']].dropna(); print('corr lr_res vs',col, round(np.corrcoef(s[col], s['lr_res'])[0,1],4), len(s))
c2 = c.reset_index(); c2['hn']=c2.h+1
mm = c2.merge(c2[['stay_id','h','lr_res']].rename(columns={'h':'hn','lr_res':'lr_next'}), on=['stay_id','hn'])
print('lag1 autocorr lr_res', round(np.corrcoef(mm.lr_res, mm.lr_next)[0,1],4))
st = c.index.get_level_values('stay_id').values
for k in [1,2,3,6]:
    y = (c[f'ma_p{k}'] - c['map_arterial']).values; ok = ~np.isnan(y)
    mz = (c['map_arterial'].values-65)/10
    Xn = np.column_stack([np.ones(len(c)), mz, np.maximum(mz,0), np.maximum(mz-1,0), np.maximum(mz+1,0), c['ne_m1'].values, np.log(c['ne_m1'].values),
                          np.nan_to_num(c['ma_m1'].values-65)/10, c['ma_m1'].isna().values, c['hh'].values/24, c['dev'].values])
    b, se, _ = ols(y[ok], Xn[ok], st[ok]); print(f'NE k={k}: n={ok.sum()}, effect per unit rate deviation {b[-1]:.3f} (se {se[-1]:.3f})')
```