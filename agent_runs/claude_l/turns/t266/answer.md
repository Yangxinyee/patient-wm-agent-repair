Run 248 failed in three ways:
- The decoder built only from features that are zero off-drug fitted the training rows, but its coefficients reached 31,000. Its linear part then overwhelmed the gates, so decoded residuals were in the thousands on drug bundles. The norepinephrine residual also leaked (up to 190) on bundles without norepinephrine.
- That decoder was worse than a plain linear one: RMSE 0.38 versus 0.22 at common rates. The rate information lies largely outside the top 28 principal components of u. The full 192-dimension linear decode from run 246 keeps it.
- `GB` holds raw DataFrames, not bundles, so the teacher-forced check crashed.

One good result: reading bundles now copy u exactly (error 3.6×10⁻¹⁵).

New design, simpler and well conditioned:
- each drug's residual = gate × (full-u linear rate decode − linear policy prediction from ctx);
- this is built from one exact gate pair per drug, at ±150 so leakage cannot reach it;
- the 30-dimension copy stays centred at u_empty.

I'll validate it on the captured and stored rows. Then I'll run it teacher-forced on the 100 tuning stays in `BUN` (held out from the fit), with gain 2, against the base.

**What would confirm it:**
- absent-row residuals of essentially 0;
- present-row errors close to the fit's RMSE of about 0.2 for insulin and 0.05 for norepinephrine;
- a small change in NLL;
- a clean lag-1 slope, which sets the gain.

```python
import numpy as np, torch, time
t0 = time.time()
K = 30; Pk = np.linalg.svd(Ud, full_matrices=False)[2][:K].T
print('copy err max K30', float(np.abs((Ud @ Pk) @ Pk.T - Ud).max()))
GATE = {}; DEC2 = {}
for j, nm in [(0, 'ins'), (1, 'ne')]:
    pres = Mall[:, j] > 0.5; y = np.where(pres, 1.0, -1.0); w = np.linalg.lstsq(X1_, y, rcond=None)[0]; s = X1_ @ w
    lo = s[pres].min(); hi = s[~pres].max(); thr = 0.5*(lo+hi); Lam = 150.0/(0.5*(lo-hi)); GATE[nm] = (w[:-1], w[-1]-thr, Lam)
    A = X1_[pres]; tg = Mall[pres, 2+j]; co = np.linalg.solve(A.T @ A + 1e-3*np.eye(193), A.T @ tg); err = A @ co - tg; DEC2[nm] = co
    bins_ = [(0, .01), (.01, 2), (2, 5), (5, 10), (10, 30)] if nm == 'ins' else [(0, .01), (.01, .4), (.4, 1), (1, 2), (2, 3.1)]
    print(nm, 'gate present max pre', round(float(Lam*(s[pres].max()-thr)), 1), 'decode on absent range', np.round([(X1_[~pres] @ co).min(), (X1_[~pres] @ co).max()], 2),
          [(b, int(m.sum()), round(float(np.sqrt(np.mean(err[m]**2))), 3), round(float(np.abs(err[m]).max()), 2)) for b in bins_ for m in [(tg >= b[0]) & (tg < b[1])] if m.sum()])
def mkW3(Gi, Gn, base):
    W = dict(base); W1 = np.zeros((192, 576)); b1_ = np.zeros(192); W2 = np.zeros((64, 192)); b2_ = np.zeros(64)
    for j in range(K):
        W1[2*j, 384:] = Pk[:, j]; b1_[2*j] = -ue_ @ Pk[:, j]; W1[2*j+1, 384:] = -Pk[:, j]; b1_[2*j+1] = ue_ @ Pk[:, j]; W2[j, 2*j] = 1; W2[j, 2*j+1] = -1
    for j, nm in [(0, 'ins'), (1, 'ne')]:
        g, gc, Lam = GATE[nm]; co = DEC2[nm]; uA = 2*K+2*j; uP = uA+1
        W1[uA, 384:] = Lam*g; b1_[uA] = Lam*gc
        W1[uP, 384:] = Lam*g - co[:192]; W1[uP, :192] = PH[nm][:-1]; b1_[uP] = Lam*gc - co[192] + PH[nm][-1]
        W2[K+j, uP] = -1; W2[K+j, uA] = 1
    Win = np.zeros((192, 32)); Win[:, :K] = Pk; Win[:, K] = 2*Gi*VD_['ins']; Win[:, K+1] = 2*Gn*VD_['ne']
    for nm, arr in [('z_post.0.weight', W1), ('z_post.0.bias', b1_), ('z_post.2.weight', W2), ('z_post.2.bias', b2_), ('z_in.weight', Win), ('z_in.bias', ue_.copy())]:
        W[nm] = torch.tensor(arr, dtype=torch.float64)
    return W
Wt = mkW3(0.0, 0.0, Wb); zc, tc = zpost_np(Wt, Xc); isr = np.array([inf is None for *_, inf in CAP])
print('CAP copy err max', float(np.abs(tc-Ucap).max()), 'reading max', float(np.abs(tc-Ucap)[isr].max()))
for j, nm in [(0, 'ins'), (1, 'ne')]:
    has = np.array([0 if inf is None else inf[1+2*j] for *_, inf in CAP]); rate = np.array([0 if inf is None else inf[2*j] for *_, inf in CAP])
    tg = np.where(has > 0.5, rate - (Xc[:, :192] @ PH[nm][:-1] + PH[nm][-1]), 0.0); d_ = np.abs(zc[:, K+j]-tg)
    print(nm, 'CAP present max err', round(float(d_[has > 0.5].max()), 4), 'rms', round(float(np.sqrt(np.mean(d_[has > 0.5]**2))), 4), 'absent max', float(d_[has < 0.5].max()))
zd, _ = zpost_np(Wt, Xd_)
for j, nm in [(0, 'ins'), (1, 'ne')]:
    has = DRm[:, j] > 0.5; tg = np.where(has, DRm[:, 2+j] - (DRc @ PH[nm][:-1] + PH[nm][-1]), 0.0); d_ = np.abs(zd[:, K+j]-tg)
    print(nm, 'DR present rms', round(float(np.sqrt(np.mean(d_[has]**2))), 4), 'max', round(float(d_[has].max()), 3), 'absent max', float(d_[~has].max()))
We = mkW3(2.0, 2.0, Wb); RZ = []
def zlog(ctx, e, u, info):
    z = bmlp(We, torch.cat([ctx, e, u], -1), 'z_post')[:, :32]; RZ.append((float(z[0, K]), float(z[0, K+1]))); return blin(We, z, 'z_in')
rows = []
for s in b1:
    B_ = BUN[s]; Rb = tf_stay_z(Wb, B_, None); RZ.clear(); Re = tf_stay_z(We, B_, zlog)
    zl = [(B_[k][0], RZ[k-1]) for k in range(1, len(B_))]
    for (t, c, mb, nb), (_, _, me, ne_) in zip(Rb, Re):
        ri = sum(a_[0] for tt, a_ in zl if t-1 < tt < t); rn = sum(a_[1] for tt, a_ in zl if t-1 < tt < t)
        rows.append((c, me-mb, ne_-nb, ri, rn))
ra = np.array(rows); print('n readings', len(ra), 'mean dNLL', round(float(ra[:, 2].mean()), 5), 'time', round(time.time()-t0, 1))
for c in sorted(set(ra[:, 0].astype(int))):
    m = ra[:, 0] == c; X_ = np.column_stack([ra[m, 3], ra[m, 4], np.ones(m.sum())]); b_ = np.linalg.lstsq(X_, ra[m, 1], rcond=None)[0]; nd = m & (ra[:, 3] == 0) & (ra[:, 4] == 0)
    print('code', c, 'n', int(m.sum()), 'dNLL', round(float(ra[m, 2].mean()), 5), 'slope on r_ins, r_ne', np.round(b_[:2], 3), 'mean|dmean| no-drug rows', round(float(np.abs(ra[nd, 1]).mean()), 5))
```