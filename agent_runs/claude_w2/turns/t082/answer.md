The grid `gg` is in "lead" form: `gf_lk` is glucose_finger k hours ahead. That is what my earlier deconvolution used. The earlier forward outputs are:
- fwd_1: dose_edit, original model, sel1;
- fwd_2: factual, sel1 (tuning);
- fwd_3: states, selT (train);
- fwd_4: factual, selT.

**This run.** I estimate the one-step structure of glucose_finger directly from the train records. I regress glucose at t on:
- glucose at t−1..t−3;
- insulin rates at t−1..t−10;
- the policy-expected rates π(latest glucose) at t−1..t−4, as controls.

With π controlled, the dose coefficients are identified by the within-band randomisation. I then iterate the fitted AR model with a single unit dose to get its implied impulse response, and compare that with my deconvolution estimate K.

**How to read the result.** The AR coefficients settle between the two explanations:
- If the glucose AR coefficients sum to about 0.8 and the dose coefficients are concentrated at lags 1–2, then glucose mean-reverts and insulin acts briefly. A lag-1/2 correction in the head would then be enough, as long as the model's own persistence matches.
- If the AR sum is about 1 and the dose coefficients turn positive at later lags, then a short-lag correction alone would overshoot.

I also peek at the factual csv layout, so I can run the same regression on the model's predicted means next.

```python
import numpy as np, pandas as pd
b0 = gg[['stay_id','h','gf','gl','ma','mc','ins0','nor0','glu_now']].sort_values(['stay_id','h']).reset_index(drop=True)
grp = b0.groupby('stay_id', sort=False)
for k_ in range(1, 4): b0[f'G{k_}'] = grp['gf'].shift(k_)
for k_ in range(1, 11): b0[f'd{k_}'] = grp['ins0'].shift(k_).fillna(0.0)
pol = np.where(b0['glu_now'].notna(), pfun(b0['glu_now'].fillna(0).values), 0.0)
b0['pol'] = pol
for k_ in range(1, 5): b0[f'p{k_}'] = b0.groupby('stay_id', sort=False)['pol'].shift(k_).fillna(0.0)
rows = b0[['gf','G1','G2','G3']].notna().all(1)
D = b0[rows]
cols_ = ['G1','G2','G3'] + [f'd{k_}' for k_ in range(1, 11)] + [f'p{k_}' for k_ in range(1, 5)]
Xm = np.column_stack([np.ones(len(D))] + [D[c_].values for c_ in cols_])
yv_ = D['gf'].values
beta, *_ = np.linalg.lstsq(Xm, yv_, rcond=None)
res0 = yv_ - Xm @ beta
s2 = res0.var(); se = np.sqrt(np.diag(np.linalg.inv(Xm.T @ Xm)) * s2)
print('n', len(D), 'resid sd', round(res0.std(), 2))
for nm_, b_, s_ in zip(['const'] + cols_, beta, se): print(f'{nm_:6s} {b_:8.3f} +- {s_:.3f}')
phi = beta[1:4]; bet = beta[4:14]
y_ = np.zeros(14)
for n_ in range(1, 14):
    y_[n_] = sum(phi[j_] * y_[n_ - 1 - j_] for j_ in range(3) if n_ - 1 - j_ >= 1) + (bet[n_ - 1] if n_ <= 10 else 0)
print('AR sum', round(phi.sum(), 3), 'implied impulse:', np.round(y_[1:13], 2))
print('K deconv     :', np.round(np.array(K, dtype=float)[:10], 2) if len(K) else None)
f4 = pd.read_csv('/work/fwd_4/factual_original.csv'); print(f4.columns.tolist(), len(f4)); print(f4.head(3).to_string())
```