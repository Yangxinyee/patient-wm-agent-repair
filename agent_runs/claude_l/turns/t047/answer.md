The decoding test gives a clear answer, and it rules out the head-level plan.
- The glucose head's view carries the insulin-dose residual only for the most recent hour, with cross-validated R² of 0.30.
- At lags 2, 3, 4 and 6 the R² is zero, and for norepinephrine even lag 1 is near zero (0.05).
- The raw lag-1 dose decodes at R² 0.82, but only because it can be predicted from the glucose history.

So the model keeps essentially no memory of how a dose differed from usual. When it predicts a reading, it cannot depend directly on doses two or more hours back, except through the readings in between. A correction added only at the value head cannot recover what the view does not contain. A real fix has to give the core state a dose memory.

Whether that memory matters depends on the true direct effect of past doses, holding the intervening readings fixed. That is what the model must reproduce for its rollouts to be right, since the rollout chains one-step predictions and that chaining is valid when doses depend only on the recorded history.

I will estimate these direct coefficients from the 30,000 training stays, for free:
1. Regress each reading on recent readings, the cross-fitted policy-expected doses (as controls), and the dose residuals at lags 1 to 12.
2. Do this for insulin on finger glucose and for norepinephrine on arterial MAP.

Large direct coefficients at lags ≥2 would confirm that a missing memory explains the weak rollouts. Coefficients near zero at lags ≥2 would point instead to a weak lag-1 response, which would need a different fix.

```python
import statsmodels.api as sm_api if False else None
nS = len(P) // 26; assert nS * 26 == len(P)
def arr(D, c, n): return D[c].values.reshape(n, 26)
stays_tr = P.index.get_level_values(0).values.reshape(nS, 26)[:, 0]
rng = np.random.RandomState(0); fold_s = rng.randint(0, 2, nS); fid = np.repeat(fold_s, 26)
Xp = FP[pcols].values
hatP = {}
for drug in ['insulin', 'norepinephrine']:
    h_ = np.full(len(P), np.nan); yv = P[drug].values
    for f_ in [0, 1]:
        trm = mtr & (fid != f_); tem = mtr & (fid == f_)
        m_ = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63).fit(Xp[trm], yv[trm])
        h_[tem] = m_.predict(Xp[tem])
    hatP[drug] = h_.reshape(nS, 26)
A = {c: arr(P, c, nS) for c in ['glucose_finger','glucose_lab','map_arterial','map_cuff','insulin','norepinephrine']}
LAST = {c: pd.DataFrame(A[c]).T.ffill().T.values for c in ['glucose_finger','glucose_lab','map_arterial','map_cuff']}
EI = np.nan_to_num(A['insulin'] - hatP['insulin']); EN = np.nan_to_num(A['norepinephrine'] - hatP['norepinephrine'])
HI = np.nan_to_num(hatP['insulin']); HN = np.nan_to_num(hatP['norepinephrine'])
def lagged(M, t_idx, L): return M[:, t_idx - L]
def direct_reg(ycode, lastcodes, E, H, E2, nl=12, t0=3):
    rows_y, rows_X = [], []
    for ti in range(t0 + 1, 26):          # column index; hour = ti-1
        y_ = A[ycode][:, ti]
        X_ = [LAST[c][:, ti-1] for c in lastcodes] + [LAST[lastcodes[0]][:, ti-2], LAST[lastcodes[0]][:, ti-3]]
        X_ += [E[:, ti-L] if ti-L >= 1 else np.zeros(nS) for L in range(1, nl+1)]
        X_ += [H[:, ti-L] if ti-L >= 1 else np.zeros(nS) for L in range(1, nl+1)]
        X_ += [E2[:, ti-L] for L in range(1, 4)]
        X_ = np.column_stack(X_); ok_ = ~np.isnan(y_) & ~np.isnan(X_).any(1)
        rows_y.append(y_[ok_]); rows_X.append(X_[ok_])
    y_ = np.concatenate(rows_y); X_ = np.column_stack([np.ones(sum(len(r) for r in rows_y)), np.concatenate(rows_X)])
    b_, *_ = np.linalg.lstsq(X_, y_, rcond=None); r_ = y_ - X_ @ b_
    cov = np.linalg.inv(X_.T @ X_) * r_.var(); se_ = np.sqrt(np.diag(cov))
    k0 = 1 + len(lastcodes) + 2
    return b_[k0:k0+nl], se_[k0:k0+nl], b_[k0+nl:k0+2*nl], b_[1:k0], len(y_), r_.std()
gi, gse, gpi, gctl, n_, rsd = direct_reg('glucose_finger', ['glucose_finger','glucose_lab','map_arterial'], EI, HI, EN)
print('INSULIN->finger direct eps coefs (lag1..12):', np.round(gi, 2)); print('   se', np.round(gse, 2)); print('   pihat coefs', np.round(gpi, 2), 'ctl', np.round(gctl, 3), 'n', n_, 'rsd', round(rsd, 2))
mi, mse, mpi, mctl, n2_, rsd2 = direct_reg('map_arterial', ['map_arterial','map_cuff','glucose_finger'], EN, HN, EI)
print('NE->art MAP direct eps coefs:', np.round(mi, 2)); print('   se', np.round(mse, 2)); print('   pihat coefs', np.round(mpi, 2), 'ctl', np.round(mctl, 3), 'n', n2_, 'rsd', round(rsd2, 2))
```