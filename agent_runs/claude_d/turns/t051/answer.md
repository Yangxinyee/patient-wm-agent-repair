The dose_edit CSV has one row per stay-hour with a `reached` flag, and the readout averages the shift over reached rows. So my targets must be computed on those exact rows.

Next, I'll re-estimate the NE effect on MAP with log-rate as the treatment. I define A_s = log(rate_s / 0.08) on running hours and 0 when off. This joins the 0.08 stopping threshold smoothly. It is an SNMM fitted as a just-identified IV regression: MAP_t on A_{t-1..t-K}, instrumented by the random titration noise EPSN_{t-1..t-K}. Under the log model, the ×2 and ×½ targets should come out nearly symmetric.

```python
K = 8
A_log = np.where(NOR>0, np.log(np.maximum(NOR,1e-3)/0.08), 0.0)
A_log = np.maximum(A_log, 0)
Zn = np.nan_to_num(EPSN)
def iv_fit(Y, A, Zi, K, tmin=None, extra_ctrl=None):
    ys=[];Xs=[];Zs=[];Cs=[]
    for t in range(K+1, Y.shape[1]):
        ok = ~np.isnan(Y[:,t])
        Xa = np.column_stack([A[ok,t-k] for k in range(1,K+1)])
        Za = np.column_stack([Zi[ok,t-k] for k in range(1,K+1)])
        Ca = np.column_stack([np.ones(ok.sum()), np.nan_to_num(Y[ok,t-K-1], nan=np.nanmean(Y[:,t-K-1])), np.isnan(Y[ok,t-K-1]), A[ok,t-K-1], np.full(ok.sum(), t)])
        ys.append(Y[ok,t]); Xs.append(Xa); Zs.append(Za); Cs.append(Ca)
    y=np.concatenate(ys); X=np.vstack(Xs); Zm=np.vstack(Zs); Cm=np.vstack(Cs)
    # partial out controls
    def res(M): return M - Cm@lstsq(Cm, M, rcond=None)[0]
    yr, Xr, Zr = res(y), res(X), res(Zm)
    b = np.linalg.solve(Zr.T@Xr, Zr.T@yr)
    u = yr - Xr@b
    # robust se
    ZX = np.linalg.inv(Zr.T@Xr)
    S = (Zr*u[:,None]).T@(Zr*u[:,None])
    V = ZX@S@ZX.T
    return b, np.sqrt(np.diag(V)), len(y)
g_log, se_log, n_ = iv_fit(MA, A_log, Zn, K)
print('n', n_); print('gamma_log', g_log.round(2)); print('se', se_log.round(2))
A_lin = NOR.copy()
g_lin, se_lin, _ = iv_fit(MA, A_lin, Zn, K)
print('gamma_lin', g_lin.round(2)); print('se', se_lin.round(2))
```