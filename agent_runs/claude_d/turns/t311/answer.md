Run 281 shows why E5 barely worked. The model encodes NE rate almost as two categories:
- doses of 0.1–0.5 sit about 19.4 away from u(1.0);
- doses of 0.6–1.0 sit within about 2 of u(1.0).

So the in-range response (including x1/2, which mostly moves patients from the high group to the low group) runs through this large low↔high contrast. E5's slope direction is only 18% aligned with it (cosine 0.18).

A representation for doses above 1.0 should therefore push u further along the low→high contrast, past u(1.0). Because u is layer-normalised, this can only be a bounded move along the sphere. That may give a partial gain, not the full +3.5.

This run finds the `mlp1` outputs that produce such "super-high" u's: targets u1 + α(u1 − u_low), solved by damped Gauss-Newton with a finite-difference Jacobian. It also checks how `aNE(r, cap)` builds the bin-16 atoms against the stored `Mo`, and how the four repurposed units' gate inputs vary with r′.

It works if the contrast gain rises clearly above 1 (e.g. 1.2–1.4) with moderate ‖m − m10‖. If the gain stays near 1, the u-space route is closed.

```python
import numpy as np, time
print([(n_, SHAPES[n_]) for n_ in NAMES if n_.startswith('encode_action') and 'mlp1' not in n_])
try:
    tb = np.asarray(u_from_m(np.stack([m10, m10+0.1*dm]))); BATCH = tb.shape == (2,192); print('batch', tb.shape)
except Exception as ex: print('batch ERR', repr(ex)); BATCH = False
def U(Ms):
    Ms = np.atleast_2d(Ms)
    if BATCH: return np.asarray(u_from_m(Ms)).reshape(len(Ms),-1)
    return np.stack([np.asarray(u_from_m(x)).ravel() for x in Ms])
cap16 = None
for c_ in [True, False]:
    try:
        a_ = np.stack([np.asarray(aNE(r_, c_)).ravel() for r_ in rp[:5]])
        e_ = float(np.abs(np.asarray(m1(a_)) - Mo[:5]).max()); print('cap', c_, 'err vs Mo', round(e_,4))
        if e_ < 1e-3: cap16 = c_
    except Exception as ex: print('cap', c_, 'ERR', repr(ex))
u1 = U(m10)[0]
ul = U(np.asarray(m1(np.stack([np.asarray(aNE(r_)).ravel() for r_ in [0.2,0.3,0.4]])))).mean(0)
dd_ = u1-ul; print('|u1|', round(float(np.linalg.norm(u1)),2), '|u1-ul|', round(float(np.linalg.norm(dd_)),2))
def invert(ut, m0, it=12, h=1e-3, lam=1e-2):
    m = m0.copy()
    for k in range(it):
        u0 = U(m)[0]; J = (U(m + h*np.eye(192)) - u0).T / h
        m = m + np.linalg.solve(J.T@J + lam*np.eye(192), J.T@(ut-u0))
    return m, float(np.linalg.norm(ut - U(m)[0]))
MX = {}
for al in [0.15, 0.3, 0.6, 1.0]:
    t0 = time.time(); mX, e_ = invert(u1 + al*dd_, m10); uX = U(mX)[0]; MX[al] = mX
    print('al', al, 'resid', round(e_,2), '|uX-u1|', round(float(np.linalg.norm(uX-u1)),2), 'gain', round(float((uX-ul)@dd_/(dd_@dd_)),3), '|dm|', round(float(np.linalg.norm(mX-m10)),1), 'max|m|', round(float(np.abs(mX).max()),1), round(time.time()-t0,1), 's')
if cap16 is not None:
    W10s = np.asarray(load('encode_action.mlp1.0.weight')); b10s = np.asarray(load('encode_action.mlp1.0.bias')); uu = [674,411,314,524]
    A16 = np.stack([np.asarray(aNE(r_, cap16)).ravel() for r_ in rp]); pre16 = A16 @ W10s[uu].T + b10s[uu]
    print('pre16 min', np.round(pre16.min(0),2), 'max', np.round(pre16.max(0),2), 'corr rp', np.round([np.corrcoef(pre16[:,j], rp)[0,1] for j in range(4)],2), 'rp range', round(float(rp.min()),2), round(float(rp.max()),2))
```