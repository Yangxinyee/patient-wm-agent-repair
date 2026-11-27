Three findings:
- The 25-component basis reproduces every grid and data u within 0.024 (norm), with coefficients up to about 10.
- All three gates are linearly separable on the full grid. "Any drug" has a margin of 2 with a tiny weight; insulin has 1.95 (|w| 56); NE has 1.46 (|w| 347, steep but valid).
- The stored rate readouts fI and fN are badly wrong on the full grid: rms 1.24 and maximum 18 for insulin, rms 0.83 and maximum 11.6 for NE. They were presumably fit on a narrower set, possibly NE only up to 1.0 rather than the 3.0 that ×3 reaches.

This matters, because the injected surprise must be exactly linear in the dose for the ×0 to ×3 edits to come out right.

Next run:
- refit the readouts separately for each presence pattern (insulin-only, NE-only, both);
- check whether the four patterns (none, I-only, N-only, both) are linearly separable from each other, so each pattern can get its own gated readout unit.

What would confirm the plan: near-exact readouts (rms below about 0.01) within each pattern, plus separable patterns. If the "both" pattern cannot be read linearly, I would restrict the exact readout to what matters most, the rates the evaluation actually produces.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
pat = np.where(hasI_g & ~hasN_g, 1, np.where(~hasI_g & hasN_g, 2, np.where(hasI_g & hasN_g, 3, 0)))
print('pattern counts', np.bincount(pat))
def rfit(X, y, alpha=1e-6):
    m = Ridge(alpha=alpha).fit(X, y); p = m.predict(X)
    return m, float(np.sqrt(((p - y)**2).mean())), float(np.abs(p - y).max())
ro = {}
for nmr, mask, yv in [('I_only', pat == 1, aG), ('N_only', pat == 2, bG), ('both_I', pat == 3, aG), ('both_N', pat == 3, bG)]:
    for al in [1e-8, 1e-4]:
        m, rms, mx = rfit(UG[mask], yv[mask], al)
        print(f'{nmr:7s} alpha {al:g}: n {mask.sum()} rms {rms:.4f} max {mx:.4f}')
    ro[nmr] = rfit(UG[mask], yv[mask], 1e-8)[0]
sepres = {}
for k in [1, 2, 3]:
    yy = np.r_[(pat == k).astype(int), 0]
    c = LinearSVC(C=1e4, max_iter=200000, tol=1e-8).fit(Xall, yy); s = c.decision_function(Xall)
    sepres[k] = (c.coef_[0].copy(), float(c.intercept_[0]))
    print(f'pattern {k} vs rest: min pos {s[yy==1].min():.3f} max neg {s[yy==0].max():.3f} |w| {np.linalg.norm(c.coef_):.1f}')
bothI = pat == 3
ai, bi = aG[bothI], bG[bothI]
print('both grid: ins rates', np.unique(ai)[:6], '... n', len(np.unique(ai)), '| NE rates n', len(np.unique(bi)), 'NE max', bi.max())
```