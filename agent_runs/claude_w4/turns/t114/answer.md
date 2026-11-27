The gates work:
- **Insulin gate:** separates cleanly on the dense grid (insulin bundles ≥0.886, others ≤−0.355).
- **NE gate:** needed a harder-margin fit (C=1000) and then separates too (NE bundles ≥0.736, others ≤−0.001).

The rate readouts produce huge values where their drug is absent (up to −870 for insulin). So the rate term must enter each unit at a small scale (0.01 for insulin, 0.05 for NE), with the gate dominating, and be scaled back up in the output weights. Then the off units stay below −16 in every case.

Now I build the first real edit in z_post with 4 repurposed hidden units, chosen as the least likely to be active anywhere. Each drug gets a pair: (gate + scaled surprise) and (gate alone). Their difference writes −κ_I·(insulin rate − E[rate|ctx]) along the glucose z-direction, and +κ_N·(NE surprise) along the MAP direction. I start with κ_I = 2 mg/dL and κ_N = 4 mmHg per unit of surprise, then calibrate with the forward tool.

Before saving, I verify on the 2319 real drug bundles from the states that the new z mean differs from the original by exactly the intended amount.

```python
clfN = LinearSVC(C=1000.0, max_iter=500000, tol=1e-8).fit(UG2, yN2)
wgN2, bgN2 = clfN.coef_.ravel(), clfN.intercept_[0]; scN2 = UG2 @ wgN2 + bgN2
thN = 0.5*(scN2[yN2].min() + scN2[~yN2].max()); thI = 0.5*(scI2[yI2].min() + scI2[~yI2].max())
mkI = ~np.isnan(RI_d); mkN = ~np.isnan(RN_d)
EI = Ridge(alpha=10.0).fit(C3[mkI], RI_d[mkI]); EN = Ridge(alpha=1.0).fit(C3[mkN], RN_d[mkN])
W2f = load('z_post.2.weight').double()
cand_units = [int(j) for j in np.asarray(cand)[:8]]
print('cand', cand_units, 'W2 col norms', np.round(W2f[:, cand_units].norm(dim=0).numpy(), 2))
def build_edit(kapI, kapN, units, cI=0.01, cN=0.05, MI=40.0, MN=60.0):
    W0n = load_original('z_post.0.weight').double().clone(); b0n = load_original('z_post.0.bias').double().clone(); W2n = load_original('z_post.2.weight').double().clone()
    jA, j0, jB, j0n = units
    t = lambda v: torch.as_tensor(np.asarray(v), dtype=torch.float64)
    for j in units: W0n[j] = 0.0; W2n[:, j] = 0.0
    W0n[j0, 384:] = MI*t(wgI); b0n[j0] = MI*(bgI - thI)
    W0n[jA, 384:] = MI*t(wgI) + cI*t(fI.coef_); W0n[jA, :192] = -cI*t(EI.coef_); b0n[jA] = MI*(bgI - thI) + cI*(fI.intercept_ - EI.intercept_)
    W0n[j0n, 384:] = MN*t(wgN2); b0n[j0n] = MN*(bgN2 - thN)
    W0n[jB, 384:] = MN*t(wgN2) + cN*t(fN.coef_); W0n[jB, :192] = -cN*t(EN.coef_); b0n[jB] = MN*(bgN2 - thN) + cN*(fN.intercept_ - EN.intercept_)
    dG = t(D4[:, 0]); dM = t(D4[:, 2])
    W2n[:32, jA] = -(kapI/cI)*dG; W2n[:32, j0] = (kapI/cI)*dG
    W2n[:32, jB] = (kapN/cN)*dM; W2n[:32, j0n] = -(kapN/cN)*dM
    return W0n, b0n, W2n
units4 = cand_units[:4]
W0n, b0n, W2n = build_edit(2.0, 4.0, units4)
b2f = load('z_post.2.bias').double()
def zmean(W0_, b0_, W2_, X): return (gel(X @ W0_.T + b0_) @ W2_.T + b2f)[:, :32]
z_old = zmean(W0f, b0f, W2f, XIN3d); z_new = zmean(W0n, b0n, W2n, XIN3d)
sI = np.where(mkI, fI.predict(np.asarray(XIN3d[:, 384:])) - EI.predict(C3), 0.0); sN = np.where(mkN, fN.predict(np.asarray(XIN3d[:, 384:])) - EN.predict(C3), 0.0)
target = -2.0*sI[:, None]*D4[:, 0][None] + 4.0*sN[:, None]*D4[:, 2][None]
err_ = (z_new - z_old).numpy() - target
print('max |dz - target| %.2e   |target| mean %.3f   old units contrib at drug bundles %.2e' % (np.abs(err_).max(), np.abs(target).mean(), float((gel(XIN3d @ W0f.T + b0f)[:, units4] @ W2f[:, units4].T).abs().max())))
print('pre-act check: min on-side gate I %.1f, max off-side unit A %.1f' % (float((XIN3d @ W0n.T + b0n)[torch.as_tensor(mkI), units4[1]].min()), float((XIN3d @ W0n.T + b0n)[torch.as_tensor(~mkI), units4[0]].max())))
```