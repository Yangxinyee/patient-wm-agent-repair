Run 195 got the token variants working. tok_with returns (row selection, tokens), and the x1 tokens match TOK exactly. Doubling insulin changes the FFN input noticeably (mean absolute change 0.069 per coordinate).

Two problems remain:
- **Undefined helpers:** gate_fit2 and the other helpers were never defined, because run 194 crashed first.
- **Unusable base_r:** it averages 0.47 on insulin-present bundles and is NaN on 250 of them, so it is not the band-mean expected rate. I will build E[rate|band] directly from band_d (glucose band at each drug bundle) and band_mean (0/1.06/2.10/3.15/4.28).

This run redefines the helpers and reruns the gate and readout fits. It also checks that "insulin present" matches band > 0, which the sliding-scale policy implies. The confirm/refute criteria are the same as before.

```python
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
t0 = time.time()
bd = np.asarray(band_d).astype(int); bmean = np.asarray(band_mean, float)
br = bmean[bd]
print("presence vs band>0 crosstab: ins&band>0 %d, ins&band0 %d, noins&band>0 %d" % ((pI_ & (bd > 0)).sum(), (pI_ & (bd == 0)).sum(), (~pI_ & (bd > 0)).sum()))
def rows(mask, v): return N3[v][mask].reshape(-1, 192)
def gate_fit2(pmask, C=1.0):
    Xp = np.concatenate([rows(pmask, v) for v in V]); Xn = np.concatenate([rows(~pmask, v) for v in V] + [n3n.reshape(-1, 192)])
    X = np.concatenate([Xp, Xn]); y = np.r_[np.ones(len(Xp)), -np.ones(len(Xn))]
    c = LinearSVC(C=C, dual=False, tol=1e-6, max_iter=20000).fit(X, y)
    w = c.coef_.ravel(); b = float(c.intercept_[0]); sp, sn = Xp @ w + b, Xn @ w + b
    return w, b, sp.min(), np.percentile(sp, 1), sn.max(), np.percentile(sn, 99), int((sp < 0).sum()), int((sn > 0).sum())
gI = gate_fit2(pI_); gN = gate_fit2(pN_)
print("gate I: on min %.3f p1 %.3f | off max %.3f p99 %.3f | errs %d %d" % gI[2:])
print("gate N: on min %.3f p1 %.3f | off max %.3f p99 %.3f | errs %d %d" % gN[2:])
kf = KFold(5, shuffle=True, random_state=0); C3 = np.asarray(CTX3, float)
idxN = np.where(pN_)[0]; cN_hat = np.zeros(len(RN))
for tr, te in kf.split(idxN):
    m_ = Ridge(alpha=1.0).fit(C3[idxN[tr]], RN[idxN[tr]]); cN_hat[idxN[te]] = m_.predict(C3[idxN[te]])
print("cN_hat R2 %.3f" % (1 - np.var(RN[pN_] - cN_hat[pN_]) / np.var(RN[pN_])))
def fit_x(pmask, rate, comp, which, alpha=1e-2):
    Xs, ys = [], []
    for v in V:
        m = v[0] if which == 'I' else v[1]
        Xs.append(rows(pmask, v)); ys.append(np.repeat(m * rate[pmask] - comp[pmask], 4))
    mdl = Ridge(alpha=alpha).fit(np.concatenate(Xs), np.concatenate(ys)); res = {}
    for v in V:
        m = v[0] if which == 'I' else v[1]
        e = rows(pmask, v) @ mdl.coef_ + mdl.intercept_ - np.repeat(m * rate[pmask] - comp[pmask], 4); res[v] = (e.mean(), np.sqrt((e**2).mean()))
    return mdl, res
mI, resI = fit_x(pI_, RI, br, 'I'); print("x_I:", " ".join("%s:%+.3f/%.3f" % (v, *resI[v]) for v in V))
mN, resN = fit_x(pN_, RN, cN_hat, 'N'); print("x_N:", " ".join("%s:%+.3f/%.3f" % (v, *resN[v]) for v in V))
e1 = (rows(pI_, (1,1)) @ mI.coef_ + mI.intercept_).reshape(-1, 4).mean(1) - (RI[pI_] - br[pI_])
G_ = G_all[pI_]
for lo, hi in [(150,180),(180,220),(220,260),(260,999)]:
    s = (G_ > lo) & (G_ <= hi); print("  G(%d,%d] n=%d resid mean %+.3f sd %.3f" % (lo, hi, s.sum(), np.nanmean(e1[s]), np.nanstd(e1[s])))
sv = RI[pI_] - br[pI_]
print("corr(resid, surprise) %.3f ; sd surprise %.3f ; sd resid %.3f" % (np.corrcoef(e1, sv)[0, 1], sv.std(), e1.std()))
print("time %.1fs" % (time.time() - t0))
```