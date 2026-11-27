The regression shows that reading surprises are encoded in the z posterior at reading bundles, but only spread thinly across many dimensions:

| Measure | Value |
|---|---|
| Total R² of z on the four surprises | 0.28 |
| Best single z dimensions | about 0.49 |
| glucose_finger coefficient vector g, norm per standard deviation of surprise | 0.34 |
| Its largest dimensions | 13, 10, 21, 30, 22 |

So a usable glucose-surprise direction g exists in z-space. I do not yet know whether the state update listens to it through `z_in`.

The cheapest way to find out is a direct probe. I will remap `z_in`'s input as z → z + κ·g·(p·z), where p is my saved drug-bundle-selective direction (`/work/v_z26.npy`). By construction p·z is about 0 at reading bundles and tracks the z26 deviation at drug bundles.

I'll choose κ so that one extra unit of insulin rate looks to the state like a reading about 3 standard deviations below prediction. This is deliberately strong so that any effect is visible. The non-symmetric map is built as a product of symmetric rank-1 'in' operations on `z_in.weight`, and I check the product numerically before using it.

This run only checks p's selectivity and builds and verifies the operations. The forward pass comes next.

```python
vz = np.load('/work/v_z26.npy'); print('v shape', vz.shape)
pz = Zp @ vz
print('p.z sd at reading bundles', pz[rd].std().round(3), ' insulin bundles', pz[hi3==1].std().round(3), ' ne bundles', pz[hn3==1].std().round(3))
gq = coef_[0].copy()
def in_ops_for(M_target_fn, g_, p_, kappa):
    Q, _ = np.linalg.qr(np.column_stack([g_, p_]))
    m = np.eye(2) + kappa * np.outer(Q.T @ g_, Q.T @ p_)
    # find SPD X with m X symmetric: m10 a + (m11-m00) b - m01 c = 0
    best = None
    for a_ in np.linspace(0.05, 5, 60):
        for b_ in np.linspace(-5, 5, 121):
            if abs(m[0,1]) < 1e-12: continue
            c_ = (m[1,0]*a_ + (m[1,1]-m[0,0])*b_) / m[0,1]
            Xm = np.array([[a_, b_],[b_, c_]])
            ev = np.linalg.eigvalsh(Xm)
            if ev.min() <= 1e-3: continue
            S1 = m @ Xm; S2 = np.linalg.inv(Xm)
            cond_ = max(np.abs(np.linalg.eigvalsh((S1+S1.T)/2)).max()/max(np.abs(np.linalg.eigvalsh((S1+S1.T)/2)).min(),1e-9), ev.max()/ev.min())
            if best is None or cond_ < best[0]: best = (cond_, S1, S2)
    cond_, S1, S2 = best
    opsl = []
    for S_ in [(best[1]+best[1].T)/2, best[2]]:
        lam, U = np.linalg.eigh(S_)
        for j in range(2): opsl.append((Q @ U[:, j], lam[j] - 1.0))
    F = np.eye(len(g_))
    for v_, a_ in opsl: F = F @ (np.eye(len(g_)) + a_ * np.outer(v_, v_) / (v_ @ v_))
    target = np.eye(len(g_)) + kappa * np.outer(g_, p_)
    return opsl, cond_, np.abs(F - target).max()
kap = -3.0 / 1.43
opsl, cond_, err_ = in_ops_for(None, gq, vz, kap)
print('cond', round(cond_,2), 'max err', err_, [round(a_,3) for _, a_ in opsl])
```