Run 45 never executed: its first line was an invalid import that I added by mistake, so nothing was tested. I am removing that line and rerunning the same analysis unchanged.

The analysis estimates, from the training stays, the direct effect of each lagged dose residual on the next reading, holding recent readings and policy-expected doses fixed. If the lag-2+ coefficients are substantial, the model's missing dose memory explains its weak rollouts. If they are near zero, the problem is a too-weak lag-1 response.

```python
nS = len(P) // 26; assert nS * 26 == len(P)
def arr(D, c, n): return D[c].values.reshape(n, 26)
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
    y_ = np.concatenate(rows_y); X_ = np.column_stack([np.ones(len(y_)), np.concatenate(rows_X)])
    b_, *_ = np.linalg.lstsq(X_, y_, rcond=None); r_ = y_ - X_ @ b_
    cov = np.linalg.inv(X_.T @ X_) * r_.var(); se_ = np.sqrt(np.diag(cov))
    k0 = 1 + len(lastcodes) + 2
    return b_[k0:k0+nl], se_[k0:k0+nl], b_[k0+nl:k0+2*nl], b_[1:k0], len(y_), r_.std()
gi, gse, gpi, gctl, n_, rsd = direct_reg('glucose_finger', ['glucose_finger','glucose_lab','map_arterial'], EI, HI, EN)
print('INSULIN->finger direct eps coefs (lag1..12):', np.round(gi, 2)); print('   se', np.round(gse, 2)); print('   pihat coefs', np.round(gpi, 2), 'ctl', np.round(gctl, 3), 'n', n_, 'rsd', round(rsd, 2))
mi, mse, mpi, mctl, n2_, rsd2 = direct_reg('map_arterial', ['map_arterial','map_cuff','glucose_finger'], EN, HN, EI)
print('NE->art MAP direct eps coefs:', np.round(mi, 2)); print('   se', np.round(mse, 2)); print('   pihat coefs', np.round(mpi, 2), 'ctl', np.round(mctl, 3), 'n', n2_, 'rsd', round(rsd2, 2))
```