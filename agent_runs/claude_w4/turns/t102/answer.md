The grid has 3206 drug bundles. Insulin rates in the records reach 11.5 units/h. The rate z-score clips at 10, which is exactly 11.5, so above that u_k cannot tell rates apart and the x3 edit on very high doses will saturate anyway.

Now the fits:
- Linear gates for "insulin present" and "NE present", reporting the margin between classes.
- Linear rate readouts for each drug on bundles where it is present, reporting the worst error in the realistic range and the extrapolated range.

```python
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
hasI = ~np.isnan(GI); hasN = ~np.isnan(GN)
def gate_fit(y):
    clf = LinearSVC(C=100.0, max_iter=200000, tol=1e-6).fit(GU, y)
    sc = GU @ clf.coef_.ravel() + clf.intercept_[0]
    return clf.coef_.ravel(), clf.intercept_[0], sc[y].min(), sc[~y].max()
wgI, bgI, minpos, maxneg = gate_fit(hasI); print('gate I: min pos %.3f max neg %.3f |w| %.2f' % (minpos, maxneg, np.linalg.norm(wgI)))
wgN, bgN, minposN, maxnegN = gate_fit(hasN); print('gate N: min pos %.3f max neg %.3f |w| %.2f' % (minposN, maxnegN, np.linalg.norm(wgN)))
def readout(mask, y, alpha=1e-4):
    m_ = Ridge(alpha=alpha).fit(GU[mask], y[mask]); p_ = m_.predict(GU[mask]); return m_, p_
for alpha in [1e-6, 1e-4, 1e-2]:
    mI_, pI_ = readout(hasI, GI, alpha); eI = pI_ - GI[hasI]
    mN_, pN_ = readout(hasN, GN, alpha); eN = pN_ - GN[hasN]
    inI = GI[hasI] <= 11.5; inN = GN[hasN] <= 1.0
    print('alpha %g  I: rms %.3f max(<=11.5) %.3f  |w| %.1f   N: rms %.4f max(<=1) %.4f max(all) %.4f |w| %.1f' % (alpha, np.sqrt((eI**2).mean()), np.abs(eI[inI]).max(), np.linalg.norm(mI_.coef_), np.sqrt((eN**2).mean()), np.abs(eN[inN]).max(), np.abs(eN).max(), np.linalg.norm(mN_.coef_)))
```