Run 245 printed the offline rollout replica. It calls the exact core step `bstep`, and that step computes the z token from the weights `z_post`/`z_in`. So any edit to those weights shows up directly in offline rollouts. Those rollouts already matched the tool's original readouts to within about 7%, row correlation 0.93. The dose conditions match the tool's four edits, and the effect estimates I fitted from the records are stored (`IRF_I`, `IRF_N`).

**Conclusion:** I can build the planned edit directly in weights and calibrate it offline before spending forward passes. The edit is a single hidden layer that computes three things:
- an exact linear copy of u, so attention to u is unchanged;
- a gate that is on only when a given drug is present;
- the dose's departure from the policy-expected dose. This is the rate decoded from u minus a linear prediction from ctx. It is added to the z token along the drug's own dose direction in token space.

This run checks the linear pieces on the stored rows:
- PCA rank of u;
- whether a linear direction separates insulin-present from insulin-absent bundles, and the same for norepinephrine;
- how well a linear map recovers the rate from u, including rescaled rates;
- how well a linear map from ctx recovers the policy-expected rate, held-out by stay.

**What would settle it:** if u has low rank, the presence gates separate with a clear margin, the rate is decoded with near-zero error, and ctx predicts the policy rate with R² above 0.7, the construction is feasible. A failure in any one of these tells me which part needs nonlinear units.

```python
import numpy as np, torch
zn = [n for n in NAMES if n.startswith('z_')]; print({n: SHAPES[n] for n in zn})
DRm = np.array([[float(r[2]), float(r[3]), r[4], r[5], r[6], r[7]] for r in DR]); DRst = np.array([r[0] for r in DR])
DRc = np.stack([np.asarray(r[8], dtype=float) for r in DR]); DRe = np.stack([np.asarray(r[9], dtype=float) for r in DR]); DRu = np.stack([np.asarray(r[10], dtype=float) for r in DR])
URm = np.array([[float(r[0]), float(r[1]), r[2], r[3]] for r in UR]); URu = np.stack([np.asarray(r[4], dtype=float) for r in UR])
ue = np.asarray(u_empty, dtype=float).ravel()
Ua = np.vstack([DRu, URu, ue[None]]); mu_u = Ua.mean(0); sv = np.linalg.svd(Ua - mu_u, full_matrices=False)[1]
ev = np.cumsum(sv**2)/np.sum(sv**2); print('PCA resid frac k=8,12,16,20,24,28,32:', [float(1-ev[k-1]) for k in [8,12,16,20,24,28,32]])
Mall = np.vstack([DRm[:, :4], URm, [[0, 0, 0, 0]]])
for j, nm in [(0, 'ins'), (1, 'ne')]:
    y = np.where(Mall[:, j] > 0.5, 1.0, -1.0); X1_ = np.hstack([Ua, np.ones((len(Ua), 1))])
    w = np.linalg.lstsq(X1_, y, rcond=None)[0]; s = X1_ @ w
    print(nm, 'present min s', s[y > 0].min(), 'absent max s', s[y < 0].max(), 'n pres', int((y > 0).sum()))
    pres = Mall[:, j] > 0.5; X2_ = X1_[pres]; rr = Mall[pres, 2 + j]
    a = np.linalg.lstsq(X2_, rr, rcond=None)[0]; err = X2_ @ a - rr
    print(nm, 'rate decode: rate range', rr.min(), rr.max(), 'rmse', np.sqrt(np.mean(err**2)), 'maxabs', np.abs(err).max())
    rowsj = DRm[:, j] > 0.5; st = DRst[rowsj]; C_ = np.hstack([DRc[rowsj], np.ones((rowsj.sum(), 1))]); ph = DRm[rowsj, 4 + j]; rt = DRm[rowsj, 2 + j]
    us = np.unique(st); rng = np.random.default_rng(0); te = np.isin(st, rng.choice(us, len(us)//4, replace=False))
    for lam in [1e-3, 1e-1, 1.0]:
        A_ = C_[~te]; bb = np.linalg.solve(A_.T @ A_ + lam*np.eye(A_.shape[1]), A_.T @ ph[~te]); p_ = C_[te] @ bb
        print(nm, 'lam', lam, 'pihat R2 vs gbm', 1-np.var(ph[te]-p_)/np.var(ph[te]), 'corr(resid_lin, resid_gbm)', np.corrcoef(rt[te]-p_, rt[te]-ph[te])[0, 1], 'sd res gbm', np.std(rt[te]-ph[te]), 'sd res lin', np.std(rt[te]-p_))
```