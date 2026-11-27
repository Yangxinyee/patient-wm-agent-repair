Now I know the helpers:
- `Up(a, b)` builds u from an insulin rate a and an NE rate b, with None meaning the drug is absent.
- `fI` and `fN` are the rate readouts from u; `EI` and `EN` are the expected-rate ridges on ctx.
- The grid covers insulin 0 to 11.5 (the z-score clip) and NE 0 to 3.0. That spans every edited dose the evaluation can produce: ×0, ×1/2, ×1.5, ×2 and ×3 of the recorded 0.5–7.5 and 0.02–1.0.

Next, the z-clone building blocks, each built on the full grid plus the real drug-bundle u's:
- a 25-component PCA basis for u;
- three linear gates on u ("any drug present", "insulin present", "NE present"), each with its worst-case margin;
- a check of the rate readouts fI and fN against the true grid rates, and of the shapes of EI and EN.

What would show the mechanism can work: gates with comfortable margins, so that M·margin is about 30 or more for exact switching, and readout errors below about 1% of the rate. Small margins would mean I need several gate units per decision.

```python
import numpy as np
from sklearn.svm import LinearSVC
UG = np.asarray(UG2, float)
hasI_g = ~np.isnan(aG); hasN_g = ~np.isnan(bG)
anyd = hasI_g | hasN_g
U_basis = np.vstack([UG[anyd], Uk])
mu_u = U_basis.mean(0)
_, sv_, Vt_ = np.linalg.svd(U_basis - mu_u, full_matrices=False)
r_pc = 25
Bpc = Vt_[:r_pc]
res = (U_basis - mu_u) - (U_basis - mu_u) @ Bpc.T @ Bpc
print('PCA r=25 resid max', np.linalg.norm(res, axis=1).max().round(4), ' coef abs max', np.abs((U_basis - mu_u) @ Bpc.T).max().round(2))
def fitgate(X, y, C=1e4):
    c = LinearSVC(C=C, max_iter=200000, tol=1e-8).fit(X, y)
    s = c.decision_function(X)
    return c.coef_[0].copy(), float(c.intercept_[0]), float(s[y == 1].min()), float(s[y == 0].max())
Xall = np.vstack([UG, uc[None, :]]); 
yany = np.r_[anyd.astype(int), 0]; yI = np.r_[hasI_g.astype(int), 0]; yN = np.r_[hasN_g.astype(int), 0]
gates = {}
for nmg, yy in [('any', yany), ('ins', yI), ('ne', yN)]:
    w_, b_, mp, mn = fitgate(Xall, yy)
    gates[nmg] = (w_, b_)
    print(f'gate {nmg}: min pos {mp:.3f} max neg {mn:.3f} margin {mp - mn:.3f} |w| {np.linalg.norm(w_):.2f}; data u scores range',
          np.round([(Uk @ w_ + b_).min(), (Uk @ w_ + b_).max()], 3))
print('fI coef', np.shape(fI.coef_), 'EI coef', np.shape(EI.coef_), 'EN coef', np.shape(EN.coef_))
pI_g = fI.predict(UG[hasI_g]); pN_g = fN.predict(UG[hasN_g])
print('fI err on grid: rms', np.sqrt(((pI_g - aG[hasI_g])**2).mean()).round(4), 'max', np.abs(pI_g - aG[hasI_g]).max().round(4),
      '| fN err: rms', np.sqrt(((pN_g - bG[hasN_g])**2).mean()).round(4), 'max', np.abs(pN_g - bG[hasN_g]).max().round(4))
```