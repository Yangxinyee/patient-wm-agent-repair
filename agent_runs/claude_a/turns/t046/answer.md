The z posterior still does not match for any bundle type. The error is the same everywhere, so the encoders differ from the description in ways I cannot pin down cheaply. I will stop trying to rebuild the encoders and the core.

My new plan uses only the heads, which I reproduce exactly:
- The glucose head will read the dose information that the state already carries, and turn it into a mean shift through a logit tilt. Two spare hidden units, gated on the code, will apply the shift to glucose_finger only. The same scheme handles norepinephrine on MAP.
- The shift must follow the within-tier deviation of the rate, not the rate itself. The model's predictions already include the average insulin effect for each tier, so reading the full rate would double-count it and spoil the factual fit.
- I will calibrate the gain with dose_edit runs.

This only works if the state carries the recent within-tier dose deviations at the moments glucose is predicted. Next I will rebuild the insulin tier means and per-hour deviations from all training stays. I will fit the direct-effect kernel by regressing glucose on its own lags, tier dummies and the lagged deviations. Then I will measure, with cross-validation by stay, how well the deviations at lags 1 to 4 can be decoded from the stored states. A decoding R² near 0.5 or above at lag 1 means the plan is feasible. Near zero means the dose has to be injected on the core side instead.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
tr_g = tr[tr.event=='glucose_finger'][['stay_id','hour','value']].copy(); tr_g['hr'] = np.round(tr_g.hour).astype(int)
tr_i = tr[tr.event=='insulin'][['stay_id','hour','rate']].copy(); tr_i['hr'] = np.floor(tr_i.hour + 1e-6).astype(int)
Gp = tr_g.pivot_table(index='stay_id', columns='hr', values='value')
Ip = tr_i.pivot_table(index='stay_id', columns='hr', values='rate').reindex(index=Gp.index, columns=Gp.columns).fillna(0.0)
def tier_of(g): return np.select([g >= 250, g >= 220, g >= 180, g >= 150], [4, 3, 2, 1], 0)
Tp = pd.DataFrame(tier_of(Gp.values), index=Gp.index, columns=Gp.columns)
rbar = {t: Ip.values[(Tp.values == t)].mean() for t in range(5)}
print('mean rate by tier', {t: round(v, 3) for t, v in rbar.items()}, 'P(ins>0|tier0)', (Ip.values[Tp.values == 0] > 0).mean())
Dp = Ip - Tp.apply(lambda col: col.map(rbar))
# regression of G(t) on G lags, tier dummies (lags 1..8), dev lags 1..8
hrs_ = [h for h in Gp.columns if h >= 8]
rowsX = []
for t in hrs_:
    feats = {'y': Gp[t]}
    for l in range(1, 4): feats[f'g{l}'] = Gp[t - l]
    for l in range(1, 9):
        feats[f'd{l}'] = Dp[t - l]
        for k in range(1, 5): feats[f't{l}_{k}'] = (Tp[t - l] == k).astype(float)
    rowsX.append(pd.DataFrame(feats))
DX = pd.concat(rowsX).dropna()
Xc = DX.drop(columns='y').values; yc = DX.y.values
coef = np.linalg.lstsq(np.c_[np.ones(len(Xc)), Xc], yc, rcond=None)[0]
cn_ = ['c'] + list(DX.drop(columns='y').columns)
print('n', len(DX), 'G lag coefs', [round(coef[cn_.index(f'g{l}')], 3) for l in range(1, 4)])
kdir = np.array([coef[cn_.index(f'd{l}')] for l in range(1, 9)]); print('direct dev kernel k_l', kdir.round(2), 'sum', kdir.sum().round(2))
# decodability of dev lags from states at glucose prediction times
gf = R[R.event == 'glucose_finger'].copy(); gf['hr'] = np.round(gf.hour).astype(int)
Xs = X[torch.tensor(gf.row_prev.values.astype(np.int64))][:, :960].numpy()
for l in [1, 2, 3, 4]:
    yd = np.array([Dp.at[s_, h_ - l] if (h_ - l) in Dp.columns else 0.0 for s_, h_ in zip(gf.stay_id, gf.hr)])
    pred = np.zeros(len(yd))
    for trn, tst in GroupKFold(5).split(Xs, yd, gf.stay_id):
        pred[tst] = Ridge(alpha=10.0).fit(Xs[trn], yd[trn]).predict(Xs[tst])
    print('lag', l, 'dev sd', yd.std().round(3), 'CV R2', round(1 - ((yd - pred)**2).mean() / yd.var(), 3))
```