The pooled IV gives a smooth, consistent insulin impulse response from both outcomes:

| lag (h) | lab | finger (IPW) |
|---|---|---|
| 1 | −2.0 | −1.9 |
| 2 | −2.4 | −2.25 |
| 3 | −2.05 | −1.96 |
| 4 | −1.6 | −1.6 |
| 5 | −1.0 | −1.2 |
| 6 | −1.0 | −0.9 |
| 7 | −0.6 | −0.5 |
| 8 | −1.0 | −0.4 |
| 9–11 | about −0.3 | about −0.25 |
| 12 | about 0 | about 0 |

The 12-hour sum is −12.4 for lab and −11.7 for finger. Standard errors are 0.13–0.31 per lag, so the two agree within noise. This peaks at lag 2 and decays gradually, a little faster than the window estimates suggested.

Next I will:
1. Combine the two by precision weighting, extend to K = 16 to check that nothing remains beyond lag 12, and keep the full covariance so I can put an error bar on the readout itself.
2. Apply the same pooled IV to norepinephrine on arterial MAP in line stays.
3. Recompute the four expected readouts on the exact sel100 rows.

With the smaller tail, I expect the insulin ×2 truth to come down from −17.5 to roughly −15.

```python
def pooled_iv_cov(Y, D, Zi, K, wts=None, Tmin=1, base=None):
    Sx, Hx = Y.shape
    ys, Ds, Zs, hs, bs, ws, ss = [], [], [], [], [], [], []
    Dp = np.concatenate([np.zeros((Sx,K)), D], axis=1); Zp = np.concatenate([np.zeros((Sx,K)), Zi], axis=1)
    for T in range(Tmin, Hx):
        ok = ~np.isnan(Y[:,T])
        if wts is not None: ok &= ~np.isnan(wts[:,T])
        ys.append(Y[ok,T]); hs.append(np.full(ok.sum(), T)); bs.append(base[ok]); ss.append(np.where(ok)[0])
        Ds.append(np.stack([Dp[ok, K+T-j] for j in range(1,K+1)],1)); Zs.append(np.stack([Zp[ok, K+T-j] for j in range(1,K+1)],1))
        ws.append(np.ones(ok.sum()) if wts is None else wts[ok,T])
    y = np.concatenate(ys); Dm = np.vstack(Ds); Zm = np.vstack(Zs); hh = np.concatenate(hs); bg = np.concatenate(bs); w = np.concatenate(ws); sid_ = np.concatenate(ss)
    C = np.column_stack([np.eye(Hx)[hh], bg, bg**2]); sw = np.sqrt(w)
    def resid(M):
        coef = np.linalg.lstsq(C*sw[:,None], (M*sw[:,None]) if M.ndim>1 else M*sw, rcond=None)[0]
        return M - C@coef
    yw = resid(y)*sw; Dw = resid(Dm)*sw[:,None]; Zw = resid(Zm)*sw[:,None]
    Pi = np.linalg.lstsq(Zw, Dw, rcond=None)[0]; PZ = Zw@Pi
    A_ = np.linalg.inv(PZ.T@PZ); b = A_ @ PZ.T @ yw
    res = yw - Dw@b
    # cluster-robust by stay
    G = pd.DataFrame(PZ*res[:,None]).groupby(sid_).sum().values
    Vc = A_ @ (G.T@G) @ A_
    return b, Vc, len(y)
K2 = 16
bl, Vl, _ = pooled_iv_cov(GL, INS, Z, K2, base=base_g)
bf, Vf, _ = pooled_iv_cov(GF, INS, Z, K2, wts=1.0/pfin, base=base_g)
print("lab K16:", bl.round(2), "\n  cluster se:", np.sqrt(np.diag(Vl)).round(2))
print("fin K16:", bf.round(2), "\n  cluster se:", np.sqrt(np.diag(Vf)).round(2))
# precision-weighted combination (assume independent)
Wl = np.linalg.inv(Vl); Wf = np.linalg.inv(Vf)
Vc_ins = np.linalg.inv(Wl+Wf); b_ins = Vc_ins @ (Wl@bl + Wf@bf)
print("combined insulin theta:", b_ins.round(2), "\n  se:", np.sqrt(np.diag(Vc_ins)).round(2), "\n  cumsum:", np.cumsum(b_ins).round(1))
# norepi pooled on arterial (line stays), baseline = mean MAP at h<=0
preM = x[x.h<=0].groupby('stay_id').map_pol.mean().reindex(x.stay_id.values.reshape(SS,25)[:,0]).values
preM = np.where(np.isnan(preM), np.nanmean(preM), preM)
bn, Vn, nn_ = pooled_iv_cov(MAPA, NR, ZR, K2, base=preM)
print("norepi -> map_arterial K16 n", nn_, ":", bn.round(2), "\n  cluster se:", np.sqrt(np.diag(Vn)).round(2), "\n  cumsum:", np.cumsum(bn).round(1))
np.savez('/work/theta.npz', b_ins=b_ins, Vc_ins=Vc_ins, bn=bn, Vn=Vn, bl=bl, bf=bf)
```