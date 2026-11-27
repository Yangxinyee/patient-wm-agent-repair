Two things from the per-reading file:
- Each row gives the observed value, predicted mean and bin NLL.
- Readings in the same bundle are all predicted from the state before the bundle. At hour −1 the finger glucose was predicted at 158 despite a lab reading of 230 at the same time, so there is no conditioning within a bundle.

Now the residual test. For each finger glucose reading at hour t, I'll attach the insulin doses at t−1 … t−6 and their shocks (dose minus expected dose given that hour's glucose). Then I'll regress the residual (observed − predicted) on the shocks, and separately on the expected-dose parts.

```python
tuP2 = tuP.sort_values(['stay_id','h']).copy()
tuP2['g_now'] = tuP2.glucose_finger.fillna(tuP2.glucose_lab)
tuP2['ins_base'] = np.where(tuP2.insulin.notna(), base_ins(tuP2.g_now.fillna(0)), 0.0)
tuP2['ins_exp'] = np.where(tuP2.insulin.notna(), tuP2.ins_base.map(lambda b: mb.get(b, b*1.05)), 0.0)
tuP2['ins0'] = tuP2.insulin.fillna(0)
tuP2['ins_shock'] = tuP2.ins0 - tuP2.ins_exp
gg = tuP2.groupby('stay_id')
for L in range(1,8):
    tuP2[f'shock_l{L}'] = gg.ins_shock.shift(L)
    tuP2[f'exp_l{L}'] = gg.ins_exp.shift(L)
    tuP2[f'ins_l{L}'] = gg.ins0.shift(L)
fg = fa[fa.event=='glucose_finger'].copy()
fg['h'] = fg.hour.astype(int)
fg = fg.merge(tuP2[['stay_id','h']+[f'shock_l{L}' for L in range(1,8)]+[f'exp_l{L}' for L in range(1,8)]+[f'ins_l{L}' for L in range(1,8)]], on=['stay_id','h'], how='left')
fg['res'] = fg.observed - fg.predicted_mean
sub = fg[(fg.h>=1)].dropna(subset=[f'shock_l{L}' for L in range(1,7)])
X = np.column_stack([np.ones(len(sub))]+[sub[f'shock_l{L}'] for L in range(1,7)]+[sub[f'exp_l{L}'] for L in range(1,4)])
r,_ = ols(sub.res, X, ['c']+[f'shock_l{L}' for L in range(1,7)]+[f'exp_l{L}' for L in range(1,4)], cluster=sub.stay_id)
print(len(sub)); print(r.round(3).to_string())
print("resid mean/sd:", sub.res.mean(), sub.res.std())
```