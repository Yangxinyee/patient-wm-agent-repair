Two problems:
1. **Band gates:** they are not linearly clean (residuals up to 1.5). Gating by last-bin band would switch units on in the wrong places, so I'm dropping it.
2. **Crash:** the per-arm print crashed again with "invalid index to scalar". One of the earlier names was overwritten, so I'll use fresh names.

What I've learned about tilting: it is only safe for small per-step shifts. The far bins are amplified in proportion to distance and the shift size. The large factual part of the readout (sd 5.7 mg/dL) is what blew up. So I'll switch to a "do no harm" readout:
- **Fit:** on arm differences, while being pushed to zero on factual rows. The factual shifts stay small, and the counterfactual per-step shifts (a few mg/dL) stay in the safe regime.
- **Implementation:** the cleanly code-gated unit (residual 0.007), with a global tilt clipped to ±60 mg/dL.

I'll scan the zero-level weight μ and the gain, reporting factual bin NLL and the open-loop response by arm, to find the largest response with NLL change ≤ ~0.

```python
ARM = kd.arm.values; TE = siD[i1] >= 50; TR = ~TE; DM = (LG[i1] - LG[i0]) @ gap
Xd_ = np.c_[dX, np.zeros(len(dX))]
Gd_ = Xd_[TR].T @ Xd_[TR] / TR.sum(); hd_ = Xd_[TR].T @ DM[TR] / TR.sum()
Gf0 = Xa[fa & (siD < 50)].T @ Xa[fa & (siD < 50)] / (fa & (siD < 50)).sum()
Tb = np.zeros(17); Tb[1:] = np.clip(c16 - cref, -60, 60) / Vbar
def build1(w_, lam, kap=4.0, G=20.0, j=jg):
    W1n, b1n, W2n, b2n, Ttn = Wv1.copy(), bv1.copy(), Wv2.copy(), bv2.copy(), Tt.copy()
    b2n += Wv2[:, j] * H0[:, j].mean()
    W1n[j] = G*dg[:192] + w_[:192]/kap; b1n[j] = G*dg[192] + w_[192]/kap; W2n[:, j] = kap*lam*Tb
    for c in [6, 10]: Ttn[c*17:(c+1)*17] -= G*kap*lam*Tb
    return W1n, b1n, W2n, b2n, Ttn
for mu in [0.03, 0.1, 0.3]:
    w0 = np.linalg.solve(Gd_ + mu*Gf0 + np.diag(np.r_[1e-3*np.ones(192), 1e-6]), hd_)
    w0[192] -= (Xa[fa] @ w0).mean()
    sF = Xa[fa] @ w0; pr_ = Xd_ @ w0
    resp = {int(a): round(float(pr_[TE & (ARM == a)].mean() / DM[TE & (ARM == a)].mean()), 2) for a in [1, 3, 4, 7]}
    line = f"mu {mu}: fact sd {sF.std():.2f} resp/tgt {resp}"
    for lam in [1.0, 2.0, 3.0]:
        E = build1(w0, lam); PN = sm(gelu(X @ E[0].T + E[1]) @ E[2].T + E[3] + E[4][cdA*17 + lbA]); pmN = pmV(PN)[0]
        dn_ = -np.log(PN[np.arange(len(PN)), bidx] + 1e-12) + np.log(P0[np.arange(len(P0)), bidx] + 1e-12)
        RAV = (pmN[i1] - pmN[i0]) - (pmA[i1] - pmA[i0])
        line += f" | L{lam}: dNLL g {dn_[f0a & gl].mean():.4f} m {dn_[f0a & ~gl].mean():.4f} x2 {RAV[TE & (ARM == 1)].mean():.2f}/{DM[TE & (ARM == 1)].mean():.2f} x0 {RAV[TE & (ARM == 3)].mean():.2f}"
    print(line)
```