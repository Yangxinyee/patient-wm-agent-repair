Line 3 failed with an impossible shape mismatch. Together with the earlier "scalar" crashes, this means some of my earlier arrays (such as `dm` or `gap`) have been overwritten, probably by helper code that writes globals.

I'll rebuild everything this test needs from `D4` and the dose grid with fresh, prefixed names. Then I'll run the planned scan: the code-gated unit with a bounded tilt, with the readout fitted on arm differences and pushed to zero on factual rows. For μ ∈ {0.03, 0.1, 0.3} and gains {1, 2, 3} I'll report factual bin NLL (glucose and MAP) and the open-loop x2 and x0 responses against target. I'm looking for a setting with ΔNLL ≤ 0 and a substantial response.

```python
import numpy as np, pandas as pd, torch, torch.nn.functional as F
Q_arm = np.asarray(D4['arm']).astype(int); Q_si = np.asarray(D4['si']).astype(int); Q_hj = np.asarray(D4['hj']).astype(int)
Q_t = np.round(np.asarray(D4['t'], dtype=float)).astype(int); Q_code = np.asarray(D4['code']).astype(int); Q_lb = np.asarray(D4['lb']).astype(int).ravel()
Q_X = np.asarray(D4['wcm'], dtype=np.float64); Q_val = np.asarray(D4['val'], dtype=float); Q_n = len(Q_X)
Q_DG = dose_stack(Rx, armsW, 'insulin'); Q_LG = np.asarray(lag_abs(Q_DG, Q_arm, Q_si, Q_t, L=4), dtype=float)
Q_gap = np.array([-1.68, -1.37, -0.40, 0.21]); Q_m = Q_LG @ Q_gap
Q_gl = np.isin(Q_code, [6, 10])
kf = pd.DataFrame(dict(si=Q_si, hj=Q_hj, arm=Q_arm, i=np.arange(Q_n)))
k0f = kf[(kf.arm == 0) & Q_gl].set_index(['si', 'hj']).i
kdf = kf[(kf.arm != 0) & Q_gl].join(k0f.rename('i0'), on=['si', 'hj']).dropna()
Q_i1, Q_i0 = kdf.i.values.astype(int), kdf.i0.values.astype(int); Q_ARM = kdf.arm.values.astype(int)
Q_dX = Q_X[Q_i1] - Q_X[Q_i0]; Q_DM = Q_m[Q_i1] - Q_m[Q_i0]; Q_TE = Q_si[Q_i1] >= 50; Q_TR = ~Q_TE
Q_W1, Q_b1, Q_W2, Q_b2, Q_T = [np.asarray(load_original(n), dtype=np.float64) for n in ['attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight','attr_heads.value_bin.2.bias','embed_transition.value_bin.weight']]
Q_gelu = lambda a: F.gelu(torch.as_tensor(a)).numpy()
def Q_sm(l): e = np.exp(l - l.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
Q_cen = np.asarray(CEN_of(Q_code), dtype=float)[:, :16]
def Q_pm(P): p = P[:, 1:]; s0 = p.sum(1); mm = (p*Q_cen).sum(1)/s0; return mm, (p*(Q_cen-mm[:, None])**2).sum(1)/s0
def Q_P(W1, b1, W2, b2, T_): return Q_sm(Q_gelu(Q_X @ W1.T + b1) @ W2.T + b2 + T_[Q_code*17 + Q_lb])
Q_H0 = Q_gelu(Q_X @ Q_W1.T + Q_b1); Q_P0 = Q_sm(Q_H0 @ Q_W2.T + Q_b2 + Q_T[Q_code*17 + Q_lb]); Q_pm0, Q_V0 = Q_pm(Q_P0)
Q_fa = (Q_arm == 0) & Q_gl & np.isfinite(Q_val); Q_f0 = (Q_arm == 0) & np.isfinite(Q_val)
Q_bidx = 1 + np.argmin(np.abs(np.nan_to_num(Q_val)[:, None] - Q_cen), 1); Q_r = np.arange(Q_n)
Q_imp = Q_H0.std(0) * np.linalg.norm(Q_W2, axis=0); Q_j = int(np.argmin(Q_imp))
Q_Xa = np.c_[Q_X, np.ones(Q_n)]; Q_tg = np.where(Q_gl, 1.0, -1.0)
Q_dg = np.linalg.solve(Q_Xa.T @ Q_Xa / Q_n + 1e-8*np.eye(193), Q_Xa.T @ Q_tg / Q_n)
Q_c16 = Q_cen[Q_code == 6][0]; Q_Vb = Q_V0[Q_fa].mean(); Q_cref = Q_pm0[Q_fa].mean()
Q_Tb = np.zeros(17); Q_Tb[1:] = np.clip(Q_c16 - Q_cref, -60, 60) / Q_Vb
print("n", Q_n, "pairs", len(Q_i1), "unit", Q_j, "gate res %.4f" % np.abs(Q_Xa @ Q_dg - Q_tg).max(), "Vb %.0f cref %.0f" % (Q_Vb, Q_cref))
Q_Xd = np.c_[Q_dX, np.zeros(len(Q_dX))]
Q_Gd = Q_Xd[Q_TR].T @ Q_Xd[Q_TR] / Q_TR.sum(); Q_hd = Q_Xd[Q_TR].T @ Q_DM[Q_TR] / Q_TR.sum()
Q_fr = Q_fa & (Q_si < 50); Q_Gf = Q_Xa[Q_fr].T @ Q_Xa[Q_fr] / Q_fr.sum()
def Q_build(w_, lam, kap=4.0, G=20.0):
    W1n, b1n, W2n, b2n, Tn = Q_W1.copy(), Q_b1.copy(), Q_W2.copy(), Q_b2.copy(), Q_T.copy()
    b2n += Q_W2[:, Q_j] * Q_H0[:, Q_j].mean()
    W1n[Q_j] = G*Q_dg[:192] + w_[:192]/kap; b1n[Q_j] = G*Q_dg[192] + w_[192]/kap; W2n[:, Q_j] = kap*lam*Q_Tb
    for c in [6, 10]: Tn[c*17:(c+1)*17] -= G*kap*lam*Q_Tb
    return W1n, b1n, W2n, b2n, Tn
for mu in [0.03, 0.1, 0.3]:
    w0 = np.linalg.solve(Q_Gd + mu*Q_Gf + np.diag(np.r_[1e-3*np.ones(192), 1e-6]), Q_hd); w0[192] -= (Q_Xa[Q_fa] @ w0).mean()
    line = f"mu {mu}: fact sd {(Q_Xa[Q_fa] @ w0).std():.2f}"
    for lam in [1.0, 2.0, 3.0]:
        PN = Q_P(*Q_build(w0, lam)); pmN = Q_pm(PN)[0]
        dn_ = -np.log(PN[Q_r, Q_bidx] + 1e-12) + np.log(Q_P0[Q_r, Q_bidx] + 1e-12)
        RAV = (pmN[Q_i1] - pmN[Q_i0]) - (Q_pm0[Q_i1] - Q_pm0[Q_i0])
        line += f" | L{lam}: dNLL g {dn_[Q_f0 & Q_gl].mean():.4f} m {dn_[Q_f0 & ~Q_gl].mean():.4f} x2 {RAV[Q_TE & (Q_ARM == 1)].mean():.2f}/{Q_DM[Q_TE & (Q_ARM == 1)].mean():.2f} x0 {RAV[Q_TE & (Q_ARM == 3)].mean():.2f}/{Q_DM[Q_TE & (Q_ARM == 3)].mean():.2f}"
    print(line)
```