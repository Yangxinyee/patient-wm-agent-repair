The panel layout is clear. Columns run from hour −2 (column = hour + 2), row order follows `adm`, and the insulin and glucose panels match the raw records. eps_ins is not simply rate ÷ base, so I'll build dose-surprise features myself.

The whole plan hinges on one question: does the head's input q carry the recent doses linearly and cleanly enough? A linear probe in the value head would read that signal. If q holds the lag-1 and lag-2 doses with good cross-validated R², and keeps a useful R² after removing what the recent readings already imply, the probe approach is viable. If the residual R² is low (under about 0.3), a probe correction would be noisy. It would then cost factual NLL, and I'd have to strengthen the dose signal upstream instead.

For each reading bundle of the 99 stays, I'll compute cross-validated ridge R² (grouped by stay) of q for the insulin and norepinephrine rates at lags 1–2. I'll do this both raw and after residualising on readings-history controls.

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
sid2i = {int(s): i for i, s in enumerate(adm.stay_id.values)}
Rr = R99.reset_index(drop=True)
idx = Rr.stay_id.map(lambda s: sid2i[int(s)]).values; hr = Rr.h.values.astype(int)
def pan(P, lag):
    c = hr - lag + 2; o = np.full(len(c), np.nan); ok = (c >= 0) & (c < P.shape[1]); o[ok] = P[idx[ok], c[ok]]; return o
dI = {l: np.nan_to_num(pan(ins, l)) for l in (1, 2, 3)}; dN = {l: np.nan_to_num(pan(ne, l)) for l in (1, 2, 3)}
ub = ~Rr.duplicated(['stay_id', 'h']).values          # one row per reading bundle (q shared)
def ctrl():
    cols = []
    for l in (1, 2, 3):
        gl_ = np.nan_to_num(pan(glast, l), nan=150); ml_ = np.nan_to_num(pan(mlast, l), nan=75)
        cols += [np.clip(gl_, 60, 400), (gl_ > 140), (gl_ > 160), (gl_ > 180), (gl_ > 220), (gl_ > 260), np.clip(ml_, 40, 120), (ml_ < 65), (ml_ > 75)]
    cols += [hr, dI[2] > 0, dI[3] > 0]
    return np.column_stack(cols).astype(float)
C = ctrl()
def resid(y, X):
    X1 = np.column_stack([np.ones(len(X)), X]); b = np.linalg.lstsq(X1, y, rcond=None)[0]; return y - X1 @ b
def cvr2_(X, y, grp):
    pr = np.zeros(len(y))
    for trn, tst in GroupKFold(5).split(X, y, grp):
        m = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X[trn], y[trn]); pr[tst] = m.predict(X[tst])
    return 1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum()
Xq = Q99[ub]; grp = Rr.stay_id.values[ub]; Cu = C[ub]
for nm, y in [('I1', dI[1]), ('I2', dI[2]), ('N1', dN[1]), ('N2', dN[2])]:
    yu = y[ub]; yr = resid(yu, Cu)
    print(nm, 'sd %.3f' % yu.std(), 'R2 raw %.3f' % cvr2_(Xq, yu, grp), '| resid sd %.3f R2 %.3f' % (yr.std(), cvr2_(Xq, yr, grp)))
print('bundles', ub.sum())
```