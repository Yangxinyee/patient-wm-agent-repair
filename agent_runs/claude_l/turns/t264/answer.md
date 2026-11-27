Run 246 confirmed the main pieces, with two problems:
- **Copy of u:** u has low rank. 30 principal components leave a residual fraction of about 7×10⁻⁶, so a near-exact copy fits in the z channel.
- **Presence gates:** each drug's presence is linearly separable from u with a clear margin.
- **Policy rate from ctx (insulin):** a linear map fits the policy rate well (R² 0.89).
- **Problem 1, rate decoding:** a linear map from u gives the insulin rate poorly at the high rates the dose edits create (max error 12 at rate 25.5). It needs nonlinear features.
- **Problem 2, policy rate from ctx (norepinephrine):** only R² 0.66. I'll measure later how much this costs in factual fit.

Next, I build the full edit:
- **Copy:** 30 exact linear copy channels, using pairs of units, since gelu(x) − gelu(−x) = x.
- **Gates:** for each drug, a gate pair that outputs the ctx-based policy prediction only when that drug is present.
- **Rate decoder:** 128 random gelu features of u for a nonlinear decode of the rate.
- **Output:** the two residuals (rate minus policy prediction) go into the last two z dimensions. z_in adds 2·G times the drug's dose direction in u-space.

I'll validate it on stored drug rows and on captured ctx/e/u from all bundle types. It should be an exact copy everywhere, decode the residual on drug bundles, and output zero residual elsewhere.

**What would refute it:** a large copy error, gate leakage on reading bundles, or decode errors well above about 0.2 at common rates.

```python
import numpy as np, torch, math, time
import torch.nn.functional as F
t0 = time.time(); rng = np.random.default_rng(1)
gelu_t = lambda x: F.gelu(torch.as_tensor(x, dtype=torch.float64)).numpy()
CAP = []
def zrec(ctx, e, u, info):
    CAP.append((ctx[0].numpy().copy(), e[0].numpy().copy(), u[0].numpy().copy(), None if info is None else (float(info['di'][0]), float(info['hi'][0]), float(info['dn'][0]), float(info['hn'][0]))))
    return ztok_orig(W0, ctx, e, u)
for s in b1[:15]: tf_stay_z(W0, BUN[s], zrec)
print('captured', len(CAP), round(time.time()-t0, 1))
K = 30; mu_u = Ua.mean(0); Uc_ = Ua - mu_u
Pk = np.linalg.svd(Uc_, full_matrices=False)[2][:K].T; Cc = Uc_ @ Pk; sd_c = Cc.std(0)
print('copy err max', np.abs(Cc @ Pk.T - Uc_).max())
X1_ = np.hstack([Ua, np.ones((len(Ua), 1))]); GATE = {}
for j, nm in [(0, 'ins'), (1, 'ne')]:
    y = np.where(Mall[:, j] > 0.5, 1.0, -1.0); w = np.linalg.lstsq(X1_, y, rcond=None)[0]; s = X1_ @ w
    lo = s[y > 0].min(); hi = s[y < 0].max(); thr = 0.5*(lo+hi); lam = 40.0/(0.5*(lo-hi)); GATE[nm] = (w[:-1], w[-1]-thr, lam)
    print(nm, 'gate pre present min/max', round(lam*(lo-thr), 1), round(lam*(s[y > 0].max()-thr), 1), 'absent max', round(lam*(hi-thr), 1))
M_ = 128; Rd = rng.normal(size=(K, M_))/np.sqrt(K)*1.5; off = rng.uniform(-2.5, 2.5, size=M_); Wrf = (Pk/sd_c) @ Rd
Fm_ = np.hstack([Cc, gelu_t(Uc_ @ Wrf + off), np.ones((len(Ua), 1))]); DEC = {}
for j, nm in [(0, 'ins'), (1, 'ne')]:
    pres = Mall[:, j] > 0.5; tgt = Mall[:, 2+j]*pres
    coef = np.linalg.solve(Fm_.T @ Fm_ + 1e-4*np.eye(Fm_.shape[1]), Fm_.T @ tgt); err = Fm_ @ coef - tgt; DEC[nm] = coef
    bins_ = [(0, .01), (.01, 2), (2, 5), (5, 10), (10, 30)] if nm == 'ins' else [(0, .01), (.01, .4), (.4, 1), (1, 2), (2, 3.1)]
    print(nm, 'absent maxabs', round(float(np.abs(err[~pres]).max()), 4), [(b, int(m.sum()), round(float(np.sqrt(np.mean(err[m]**2))), 3), round(float(np.abs(err[m]).max()), 2)) for b in bins_ for m in [pres & (tgt >= b[0]) & (tgt < b[1])] if m.sum()])
PH = {}; VD_ = {}
for j, nm, cap in [(0, 'ins', 6.0), (1, 'ne', 1.2)]:
    rows = DRm[:, j] > 0.5; C_ = np.hstack([DRc[rows], np.ones((rows.sum(), 1))])
    PH[nm] = np.linalg.solve(C_.T @ C_ + 1e-3*np.eye(193), C_.T @ DRm[rows, 4+j])
    m = (Mall[:, j] > 0.5) & (Mall[:, 1-j] < 0.5) & (Mall[:, 2+j] <= cap)
    A = np.hstack([Mall[m, 2+j][:, None], np.ones((m.sum(), 1))]); co = np.linalg.lstsq(A, Uc_[m], rcond=None)[0]; VD_[nm] = co[0]
    print(nm, '|v|', round(float(np.linalg.norm(co[0])), 3), 'lin R2', round(float(1-np.sum((Uc_[m]-A @ co)**2)/np.sum((Uc_[m]-Uc_[m].mean(0))**2)), 3))
def mkW(Gi, Gn, base=None):
    W = dict(W0 if base is None else base); W1 = np.zeros((192, 576)); b1_ = np.zeros(192); W2 = np.zeros((64, 192)); b2_ = np.zeros(64)
    for j in range(K):
        W1[2*j, 384:] = Pk[:, j]; b1_[2*j] = -mu_u @ Pk[:, j]; W1[2*j+1, 384:] = -Pk[:, j]; b1_[2*j+1] = mu_u @ Pk[:, j]; W2[j, 2*j] = 1; W2[j, 2*j+1] = -1
    W1[64:, 384:] = Wrf.T; b1_[64:] = off - mu_u @ Wrf
    for j, nm in [(0, 'ins'), (1, 'ne')]:
        gw, gc, lam = GATE[nm]; uA = 60+2*j; uP = 61+2*j
        W1[uA, 384:] = lam*gw; b1_[uA] = lam*gc; W1[uP, 384:] = lam*gw; b1_[uP] = lam*gc + PH[nm][-1]; W1[uP, :192] = PH[nm][:-1]
        co = DEC[nm]; row = K+j; W2[row, 0:2*K:2] += co[:K]; W2[row, 1:2*K:2] -= co[:K]; W2[row, 64:] = co[K:K+M_]; b2_[row] = co[-1]; W2[row, uP] -= 1; W2[row, uA] += 1
    Win = np.zeros((192, 32)); Win[:, :K] = Pk; Win[:, K] = 2*Gi*VD_['ins']; Win[:, K+1] = 2*Gn*VD_['ne']
    for nm, arr in [('z_post.0.weight', W1), ('z_post.0.bias', b1_), ('z_post.2.weight', W2), ('z_post.2.bias', b2_), ('z_in.weight', Win), ('z_in.bias', mu_u.copy())]:
        W[nm] = torch.tensor(arr, dtype=torch.float64)
    return W
def zpost_np(W, X):
    hh = gelu_t(X @ W['z_post.0.weight'].numpy().T + W['z_post.0.bias'].numpy()); z = hh @ W['z_post.2.weight'].numpy().T + W['z_post.2.bias'].numpy()
    return z[:, :32], z[:, :32] @ W['z_in.weight'].numpy().T + W['z_in.bias'].numpy()
Wt = mkW(0.0, 0.0)
Xc = np.stack([np.concatenate([c, e, u]) for c, e, u, _ in CAP]); Ucap = np.stack([u for _, _, u, _ in CAP]); zc, tc = zpost_np(Wt, Xc)
print('CAP copy token err max', np.abs(tc-Ucap).max(), 'rms', np.sqrt(np.mean((tc-Ucap)**2)), 'u rms', np.sqrt(np.mean(Ucap**2)))
for j, nm in [(0, 'ins'), (1, 'ne')]:
    has = np.array([0 if inf is None else inf[1+2*j] for *_, inf in CAP]); rate = np.array([0 if inf is None else inf[2*j] for *_, inf in CAP])
    phl = Xc[:, :192] @ PH[nm][:-1] + PH[nm][-1]; tg = np.where(has > 0.5, rate-phl, 0.0); d_ = np.abs(zc[:, K+j]-tg)
    print(nm, 'CAP n has', int(has.sum()), 'max err present', round(float(d_[has > 0.5].max()), 4) if has.sum() else None, 'absent max', round(float(d_[has < 0.5].max()), 5), 'phl range absent rows', round(float(phl[has < 0.5].min()), 2), round(float(phl[has < 0.5].max()), 2))
Xd_ = np.hstack([DRc, DRe, DRu]); zd, _ = zpost_np(Wt, Xd_)
for j, nm in [(0, 'ins'), (1, 'ne')]:
    has = DRm[:, j] > 0.5; tg = np.where(has, DRm[:, 2+j] - (DRc @ PH[nm][:-1] + PH[nm][-1]), 0.0); d_ = np.abs(zd[:, K+j]-tg)
    print(nm, 'DR max err present', round(float(d_[has].max()), 4), 'rms', round(float(np.sqrt(np.mean(d_[has]**2))), 4), 'absent max', round(float(d_[~has].max()), 5))
print('time', round(time.time()-t0, 1))
```