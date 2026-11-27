The insulin dose–response is linear across the observed range. The quadratic terms are small and change sign across bands (−0.19, −0.16, +0.08, +0.02), and the pooled curvature is −0.027 ± 0.018. The raw means fall almost in a straight line from 1 to 5 units in band 2 and from 1.5 to 6.5 units in band 3. Doubling mostly stays inside doses the jitter already produces, so I will treat the insulin effect as linear. That makes ×½ about −½ of ×2.

The largest remaining uncertainty is the tail of the impulse response, lags 5–12, which drives the 12-hour readout. My window estimates there were noisy and varied by window. I will switch to a pooled distributed-lag IV over every outcome hour:
- outcome: lab glucose at hour T (randomly sampled);
- regressors: doses at T−1 … T−12, each instrumented by its own jitter;
- controls: hour fixed effects and a pre-treatment baseline glucose (the reading at hour ≤ 0) to absorb between-stay variance.

As a check I will repeat it on finger readings with inverse-probability weights for the known schedule. If the two outcomes agree on the cumulative partial sums, I can trust the tail.

```python
K = 12
# baseline glucose: mean of readings at h<=0 (pre-treatment)
pre = pn[(pn.h<=0)].groupby('stay_id')[['glucose_finger','glucose_lab']].mean().mean(axis=1)
base_g = pre.reindex(sids).values
base_g = np.where(np.isnan(base_g), np.nanmean(base_g), base_g)
def pooled_iv(Y, D, Zi, K, wts=None, Tmin=1, extra_ctrl=None):
    Sx, Hx = Y.shape
    ys, Ds, Zs, hs, bs, ws = [], [], [], [], [], []
    Dp = np.concatenate([np.zeros((Sx,K)), D], axis=1); Zp = np.concatenate([np.zeros((Sx,K)), Zi], axis=1)
    for T in range(Tmin, Hx):
        ok = ~np.isnan(Y[:,T])
        if wts is not None: ok &= ~np.isnan(wts[:,T])
        ys.append(Y[ok,T]); hs.append(np.full(ok.sum(), T)); bs.append(base_g[ok])
        Ds.append(np.stack([Dp[ok, K+T-j] for j in range(1,K+1)],1)); Zs.append(np.stack([Zp[ok, K+T-j] for j in range(1,K+1)],1))
        ws.append(np.ones(ok.sum()) if wts is None else wts[ok,T])
    y = np.concatenate(ys); Dm = np.vstack(Ds); Zm = np.vstack(Zs); hh = np.concatenate(hs); bg = np.concatenate(bs); w = np.concatenate(ws)
    C = np.column_stack([np.eye(25)[hh], bg, bg**2])   # hour FE + baseline
    sw = np.sqrt(w)
    def resid(M):
        coef = np.linalg.lstsq(C*sw[:,None], M*sw[:,None] if M.ndim>1 else M*sw, rcond=None)[0]
        return M - C@coef
    yr = resid(y); Dr = resid(Dm); Zr = resid(Zm)
    Zw = Zr*sw[:,None]; Dw = Dr*sw[:,None]; yw = yr*sw
    PZ = Zw @ np.linalg.lstsq(Zw, Dw, rcond=None)[0]
    b = np.linalg.lstsq(PZ, yw, rcond=None)[0]
    res = yw - Dw@b; Vv = np.linalg.inv(PZ.T@PZ)*res.var()
    return b, np.sqrt(np.diag(Vv)), len(y)
b_lab, se_lab, n_lab = pooled_iv(GL, INS, Z, K)
print("LAB  n", n_lab); print("  theta:", b_lab.round(2)); print("  se   :", se_lab.round(2)); print("  cumsum:", np.cumsum(b_lab).round(1))
# finger with IPW: p=1 if insulin given previous hour (i.e., pol_{T-1}>150) else 0.30
INSprev = np.concatenate([np.zeros((S,1)), INS[:,:-1]], axis=1)
pfin = np.where(INSprev>0, 1.0, 0.30)
b_f, se_f, n_f = pooled_iv(GF, INS, Z, K, wts=1.0/pfin)
print("FINGER(IPW) n", n_f); print("  theta:", b_f.round(2)); print("  se   :", se_f.round(2)); print("  cumsum:", np.cumsum(b_f).round(1))
```