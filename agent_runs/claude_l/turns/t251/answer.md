The collection worked: 300 training stays and 37,844 base rows in 23 s. The u built for branches matches the replica's exactly, the bundle times are in hours, and every infusion has a π̂. Collecting more stays later is cheap.

Next I fit the decoder by ridge regression, holding out 25% of stays, in two versions:
1. **Linear in the base.** It costs only two linear-regime units of z_prior.
2. **Linear plus about 180 fixed random gelu features** of the whitened base. These are what z_prior's hidden layer can hold.

I'll report held-out R² for the insulin and NE residuals on factual rows after a drug bundle and on edited rows by factor, plus the spread of predictions on rows with no preceding drug, which should be near zero.

Hypothesis: the residual is well decodable, with R² ≥ 0.7 and gains near 1 across factors. Low R², or decoded gains that shrink at large factors, would mean the base carries the dose poorly and the head-side lever needs its signal from z_post instead.

```python
import numpy as np, torch
Xb = np.stack([r[7] for r in DEC]).astype(np.float64); sidA = np.array([r[0] for r in DEC]); aft = np.array([r[2] for r in DEC])
yI = np.array([r[3] for r in DEC]); yN = np.array([r[4] for r in DEC]); fA = np.array([r[5] for r in DEC]); dcA = np.array([r[6] for r in DEC])
ust = np.unique(sidA); te_s = set(ust[:75]); teM = np.isin(sidA, list(te_s)); trM = ~teM
mu_b = Xb[trM].mean(0); Cb = np.cov(Xb[trM].T); ev_, evec_ = np.linalg.eigh(Cb + 1e-6*np.eye(192))
Wh = evec_ / np.sqrt(ev_)          # whitening
Z = (Xb - mu_b) @ Wh
rng2 = np.random.default_rng(7); NF = 180
A_rf = rng2.normal(size=(192, NF)) / np.sqrt(192) * 1.0; b_rf = rng2.uniform(-1.5, 1.5, NF)
def gelu_np(x): return 0.5*x*(1+scipy.special.erf(x/np.sqrt(2)))
import scipy.special
Frf = gelu_np(Z @ A_rf + b_rf)
def fitr(X, y, lam):
    Xa = np.c_[X, np.ones(len(X))]; A = Xa[trM].T @ Xa[trM] + lam*np.eye(Xa.shape[1]); A[-1, -1] -= lam
    w = np.linalg.solve(A, Xa[trM].T @ y[trM]); return w, Xa @ w
def report(nm_, p, y, which):
    m1 = teM & (aft==1) & (fA==1) & (np.abs(y) > 0)
    s = 'fact R2 %.3f' % (1-np.var(y[m1]-p[m1])/np.var(y[m1]))
    for f in [0.0, 0.5, 2.0, 3.0]:
        mf = teM & (aft==1) & (fA==f) & (dcA==which)
        if mf.sum() > 10:
            sl = np.polyfit(y[mf], p[mf], 1)[0]; s += ' | f%.1f R2 %.2f slope %.2f' % (f, 1-np.var(y[mf]-p[mf])/np.var(y[mf]), sl)
    m0 = teM & (aft==0); s += ' | no-drug pred sd %.3f (target sd fact %.3f)' % (np.std(p[m0]), np.std(y[m1]))
    print(nm_, s)
for nm_, X in [('lin', Z), ('lin+rf', np.c_[Z, Frf])]:
    for lab_, y, which in [('ins', yI, 8), ('ne', yN, 9)]:
        w, p = fitr(X, y, 1.0); report(nm_+' '+lab_, p, y, which)
```