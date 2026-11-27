Run 247 showed the edit is close but has two leaks that would disturb every factual prediction:
- **Copy error on reading bundles.** The token copy had a max error of 0.12. That error is on u_empty, the u value of reading bundles. It was only one row in the PCA, so it was badly reconstructed. Reading bundles are most bundles, and one attention head attends to that token.
- **Residual leakage where the drug is absent.** The decoded residual leaked up to 0.36 on bundles without that drug. This happened because the random-feature decoder is not gated.

The fixes:
- Centre the PCA at u_empty, so reading bundles copy exactly.
- Build the rate decoder only from features that are exactly zero wherever the drug is absent. Each is a gelu ridge whose input stays at or below −8 on every absent row, so it can only be active where the drug is present.
- Pass the linear part and the policy prediction from ctx through a stronger gate, ±100 at the margin. On reading-bundle ctx the ctx-map reaches 42, so ±40 was not enough.

Then I'll check the edit teacher-forced on 150 held-out training stays, with gain 2 for both drugs, against the current base (z_head = 0). I want three numbers:
- the change in value NLL by reading code;
- the next reading's mean shift per unit residual, which calibrates the gain;
- that readings with no preceding drug are unaffected.

**What would confirm it:** zero error on absent rows, a small NLL change, and a clean linear lag-1 slope.

```python
import numpy as np, torch, time
t0 = time.time(); rng = np.random.default_rng(2)
ue_ = ue.copy(); Ud = Ua - ue_; K = 28
Pk = np.linalg.svd(Ud, full_matrices=False)[2][:K].T; Cd = Ud @ Pk
print('copy err max', float(np.abs(Cd @ Pk.T - Ud).max()), 'CODE_ID', CODE_ID)
X1_ = np.hstack([Ua, np.ones((len(Ua), 1))]); GATE = {}; FEA = {}; DEC = {}
for j, nm in [(0, 'ins'), (1, 'ne')]:
    pres = Mall[:, j] > 0.5; y = np.where(pres, 1.0, -1.0); w = np.linalg.lstsq(X1_, y, rcond=None)[0]; s = X1_ @ w
    lo = s[pres].min(); hi = s[~pres].max(); thr = 0.5*(lo+hi); Lam = 100.0/(0.5*(lo-hi)); GATE[nm] = (w[:-1], w[-1]-thr, Lam)
    sdp = Cd[pres].std(0)+1e-9; nf = 64
    Pm = (Pk/sdp) @ rng.normal(size=(K, nf))*rng.uniform(0.3, 1.5, size=nf) + np.outer(w[:-1], rng.uniform(2, 15, size=nf)/(lo-hi))
    pr_ = Ua @ Pm; qm = -8.0 - pr_[~pres].max(0); Fq = gelu_t(pr_+qm)
    A = np.hstack([Cd, Fq, np.ones((len(Ua), 1))])[pres]; tg = Mall[pres, 2+j]
    co = np.linalg.lstsq(A, tg, rcond=1e-9)[0]; err = A @ co - tg; FEA[nm] = (Pm, qm); DEC[nm] = co
    bins_ = [(0, .01), (.01, 2), (2, 5), (5, 10), (10, 30)] if nm == 'ins' else [(0, .01), (.01, .4), (.4, 1), (1, 2), (2, 3.1)]
    print(nm, 'feat absent max', float(np.abs(Fq[~pres]).max()), 'max|co|', round(float(np.abs(co).max()), 1), [(b, int(m.sum()), round(float(np.sqrt(np.mean(err[m]**2))), 3), round(float(np.abs(err[m]).max()), 2)) for b in bins_ for m in [(tg >= b[0]) & (tg < b[1])] if m.sum()])
def mkW2(Gi, Gn, base):
    W = dict(base); W1 = np.zeros((192, 576)); b1_ = np.zeros(192); W2 = np.zeros((64, 192)); b2_ = np.zeros(64)
    for j in range(K):
        W1[2*j, 384:] = Pk[:, j]; b1_[2*j] = -ue_ @ Pk[:, j]; W1[2*j+1, 384:] = -Pk[:, j]; b1_[2*j+1] = ue_ @ Pk[:, j]; W2[j, 2*j] = 1; W2[j, 2*j+1] = -1
    for j, nm in [(0, 'ins'), (1, 'ne')]:
        g, gc, Lam = GATE[nm]; co = DEC[nm]; a_u = Pk @ co[:K]; c0 = co[-1] - a_u @ ue_; uA = 2*K+2*j; uP = uA+1
        W1[uA, 384:] = Lam*g; b1_[uA] = Lam*gc
        W1[uP, 384:] = Lam*g - a_u; W1[uP, :192] = PH[nm][:-1]; b1_[uP] = Lam*gc + PH[nm][-1] - c0
        Pm, qm = FEA[nm]; f0 = 2*K+4+64*j; W1[f0:f0+64, 384:] = Pm.T; b1_[f0:f0+64] = qm
        row = K+j; W2[row, f0:f0+64] = co[K:K+64]; W2[row, uP] = -1; W2[row, uA] = 1
    Win = np.zeros((192, 32)); Win[:, :K] = Pk; Win[:, K] = 2*Gi*VD_['ins']; Win[:, K+1] = 2*Gn*VD_['ne']
    for nm, arr in [('z_post.0.weight', W1), ('z_post.0.bias', b1_), ('z_post.2.weight', W2), ('z_post.2.bias', b2_), ('z_in.weight', Win), ('z_in.bias', ue_.copy())]:
        W[nm] = torch.tensor(arr, dtype=torch.float64)
    return W
Wb = dict(W0); Wb['z_head.weight'] = torch.zeros_like(W0['z_head.weight'])
Wt = mkW2(0.0, 0.0, Wb); zc, tc = zpost_np(Wt, Xc)
print('CAP copy token err max', float(np.abs(tc-Ucap).max()), 'reading-bundle max', float(np.abs(tc-Ucap)[[inf is None for *_, inf in CAP]].max()))
for j, nm in [(0, 'ins'), (1, 'ne')]:
    has = np.array([0 if inf is None else inf[1+2*j] for *_, inf in CAP]); rate = np.array([0 if inf is None else inf[2*j] for *_, inf in CAP])
    tg = np.where(has > 0.5, rate - (Xc[:, :192] @ PH[nm][:-1] + PH[nm][-1]), 0.0); d_ = np.abs(zc[:, K+j]-tg)
    print(nm, 'CAP max err present', round(float(d_[has > 0.5].max()), 4), 'absent max', float(d_[has < 0.5].max()))
zd, _ = zpost_np(Wt, Xd_)
for j, nm in [(0, 'ins'), (1, 'ne')]:
    has = DRm[:, j] > 0.5; tg = np.where(has, DRm[:, 2+j] - (DRc @ PH[nm][:-1] + PH[nm][-1]), 0.0); d_ = np.abs(zd[:, K+j]-tg)
    print(nm, 'DR err present rms', round(float(np.sqrt(np.mean(d_[has]**2))), 4), 'max', round(float(d_[has].max()), 3), 'absent max', float(d_[~has].max()))
We = mkW2(2.0, 2.0, Wb); RZ = []
def zlog(ctx, e, u, info):
    z = bmlp(We, torch.cat([ctx, e, u], -1), 'z_post')[:, :32]; RZ.append((float(z[0, K]), float(z[0, K+1]))); return blin(We, z, 'z_in')
held = [s for s in all_ids if s not in S2][:150]; print('GB type', type(GB[held[0]]))
rows = []
for s in held:
    B_ = GB[s]; Rb = tf_stay_z(Wb, B_, None); RZ.clear(); Re = tf_stay_z(We, B_, zlog)
    tb = [B_[k][0] for k in range(1, len(B_))]; zmap = {round(t, 3): rz for t, rz in zip(tb, RZ)}
    for (t, c, mb, nb), (_, _, me, ne_) in zip(Rb, Re):
        ri = rn = 0.0
        for tt, (a1_, a2_) in zmap.items():
            if t-1 < tt < t: ri += a1_; rn += a2_
        rows.append((c, me-mb, ne_-nb, ri, rn))
ra = np.array(rows); print('n readings', len(ra), 'mean dNLL', ra[:, 2].mean(), 'time', round(time.time()-t0, 1))
for c in sorted(set(ra[:, 0].astype(int))):
    m = ra[:, 0] == c; X_ = np.column_stack([ra[m, 3], ra[m, 4], np.ones(m.sum())]); b_ = np.linalg.lstsq(X_, ra[m, 1], rcond=None)[0]
    print('code', c, 'n', int(m.sum()), 'dNLL', round(float(ra[m, 2].mean()), 5), 'slope dmean on r_ins, r_ne', np.round(b_[:2], 3), 'mean|dmean| where no drug', round(float(np.abs(ra[m & (ra[:, 3] == 0) & (ra[:, 4] == 0), 1]).mean()), 5))
```