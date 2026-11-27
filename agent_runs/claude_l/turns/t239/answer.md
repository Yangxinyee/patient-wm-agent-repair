DA_meta covers the 57,239 drug bundles of 3,114 training stays. Its columns are stay, time, insulin rate (up to 11.5 U/h), NE rate (up to 1.0), has-insulin and has-NE. DA_ctx and DA_u are the original model's z_post inputs at those bundles, so I can fit the z_post decoders by regression on them.

This run checks the two decoders the edit needs:
1. **Rate from u.** It must hold not only for recorded rates but for edited ones (x0, x1/2, x1.5, x2, x3). The rollout feeds edited atoms through `encode_action`, and insulin bins saturate at 5.0, so I build synthetic edited u's from the tuning batch's bundles exactly as `u_for` does.
2. **Expected rate from ctx (π̂).** This is the policy part that has to be subtracted. R² near my policy GBM's (insulin 0.82–0.87, NE 0.93–0.94) would confirm that ctx carries the policy information.

If the u-decoder breaks down for edited rates, I'll need a nonlinear decoder made of several gelu units. If ctx gives a much lower R², the residual would carry confounded policy signal and cost factual NLL.

```python
import numpy as np, torch
M_ = np.asarray(DA_meta); C_ = np.asarray(DA_ctx, dtype=np.float64); U_a = np.asarray(DA_u, dtype=np.float64)
tr = M_[:,0] < 3000; te = ~tr
def ridge(X, y, lam=1e-3):
    Xa = np.c_[X, np.ones(len(X))]; A = Xa.T @ Xa + lam*np.eye(Xa.shape[1]); A[-1,-1] -= lam
    return np.linalg.solve(A, Xa.T @ y)
def pr(X, w): return np.c_[X, np.ones(len(X))] @ w
for nm_, col, hcol in [('ins',2,4), ('ne',3,5)]:
    m = M_[:,hcol] > 0
    for src, X in [('u', U_a), ('ctx', C_)]:
        w = ridge(X[m&tr], M_[m&tr,col]); y = M_[m&te,col]; p = pr(X[m&te], w)
        print(nm_, src, 'test R2 %.3f resid sd %.3f (y sd %.3f)' % (1-np.var(y-p)/np.var(y), np.std(y-p), np.std(y)))
# synthetic edited u's
SY = []
for s in list(BUN)[:40]:
    for (t, atoms) in BUN[s]:
        ua = [a for a in atoms if a['act'] and a['kind']==0]
        if not ua: continue
        base_r = {}
        for a in ua:
            nmd = DRUGN[a['code']]; base_r[a['code']] = a['num'][4]*NUMV[(nmd,'rate')]['iqr'] + NUMV[(nmd,'rate')]['median']
        for dc in base_r:
            for f in [0, 0.5, 1, 1.5, 2, 3]:
                ub = [edit_atom(a, f) if (a['code']==dc and f!=1) else a for a in ua]
                u_ = set_enc(W0, [fix_atom2(a) for a in ub], 'encode_action').numpy()
                ri = base_r.get(8, 0)*(f if dc==8 else 1); rn = base_r.get(9, 0)*(f if dc==9 else 1)
                SY.append((dc, f, ri, rn, 8 in base_r, 9 in base_r, u_))
SYm = np.array([x[:6] for x in SY], dtype=float); SYu = np.stack([x[6] for x in SY]).astype(np.float64)
print('synthetic n', len(SY), 'ins rate max %.1f' % SYm[:,2].max())
for nm_, col, hcol in [('ins',2,4), ('ne',3,5)]:
    m = SYm[:,hcol] > 0
    w = ridge(SYu[m], SYm[m,col], 1e-4); p = pr(SYu[m], w); y = SYm[m,col]
    print(nm_, 'u-decode all factors R2 %.4f rmse %.3f' % (1-np.var(y-p)/np.var(y), np.sqrt(np.mean((y-p)**2))))
    for f in [0, 0.5, 1, 2, 3]:
        mf = m & (SYm[:,1]==f) & (SYm[:,0]==(8 if nm_=='ins' else 9))
        print('   f', f, 'bias %.3f rmse %.3f n %d' % (np.mean(p[mf[m]]-y[mf[m]]), np.sqrt(np.mean((p[mf[m]]-y[mf[m]])**2)), mf.sum()))
```