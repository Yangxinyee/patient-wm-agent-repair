The sweep with head-input changes is stored. Now I can test whether a relay can carry the NE dose to the head cleanly. For each head-2 basis direction I estimate its consistent per-unit-dose response vector in head-input space, from arterial queries with no recent insulin and with a lag-1 NE dose. I then compare it with the natural spread of the head input.

The best achievable signal-to-noise per unit injection norm comes from the top singular value of Σ^(−1/2)G, where G stacks the response vectors and Σ is the natural covariance. I also report the residual heterogeneity. If an injection of natural size (|α| about 25–50, similar to the model's own NE value slope of 25) reaches a dose-unit noise well below the innovation sd of 0.079, the relay is viable for NE.

```python
def relay_snr(mask, lagcol, K=4):
    q = np.where(mask)[0]; X = lagcol[q]                 # (n, 4) dose lags of the drug
    Xn = np.column_stack([X, np.ones(len(q))])
    G = np.zeros((48, 192)); Hres = 0.
    for i in range(48):
        Y = DWC[i, q, :].astype(np.float64); b = np.linalg.lstsq(Xn, Y, rcond=None)[0]; G[i] = b[0]
    return G, q
for nm, mask, lagc, cm in [('NE->art', am & noI & (LN_[:,0]>0), LN_, am), ('INS->finger', fm & noN & (LI[:,0]>0), LI, fm)]:
    G, q = relay_snr(mask, lagc)
    S = np.cov(WC0[cm].T) + 1e-6*np.eye(192); ev, U_ = np.linalg.eigh(S); Sih = U_ @ np.diag(ev**-0.5) @ U_.T
    sv = np.linalg.svd(Sih @ G.T, compute_uv=True); s_top = sv[1][0]
    a = sv[2][0]; g = G.T @ a; rho = np.linalg.solve(S, g)          # best readout for top alpha
    sig = DWC[:, q, :].astype(np.float64).transpose(1,2,0) @ a @ rho  # per-query signal per unit dose-scale
    lag1 = lagc[q,0]; slope = (sig*lag1).sum()/(lag1**2).sum(); het = np.std(sig - slope*lag1)
    nat = np.sqrt(rho @ S @ rho)
    print(nm, 'n', len(q), 'top SNR per unit|a| %.4f (2nd %.4f) | ev range %.1e..%.1e' % (s_top, sv[1][1], ev.min(), ev.max()))
    print('   readout: slope %.4f het sd %.4f nat sd %.4f -> at |a|=50: dose-noise %.4f, het/slope %.3f' % (slope, het, nat, nat/(50*abs(slope)), het/abs(slope)))
```