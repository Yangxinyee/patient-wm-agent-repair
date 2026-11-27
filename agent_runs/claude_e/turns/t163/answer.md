W2 carries a_ne·tm and a_ins·tg exactly, so the unit-gain scaling must already sit inside the readout weights (the fitted readout divided by its gain). Either way, the factual cost of an edit is the readout noise minus whatever true error it corrects.

Next I'll fit a new NE readout of the current rate level (rne1). It is a generalised ridge on per-draw head inputs, fit on map_arterial rows, with a penalty that keeps it near 0 on other rows so the gate can switch it off cleanly.

To see whether a level correction helps or hurts the factual fit, I'll regress the original model's MAP residual (observed minus predicted mean) on the unit-gain readout. That gives the NLL-optimal coefficient. A clearly positive value near 2-3 would mean the correction also improves the factual fit; near zero would mean it is pure cost. I'll compute the same check for the insulin readout ROi5 and the old NE readout ROn4.

```python
def fit_lin(tmask, y, omask, lam_o=0.01, alpha=1e-3, ndr=2, seed=0):
    rng = np.random.default_rng(seed); dr = rng.choice(16, ndr, replace=False)
    it = np.where(tmask)[0]; io = np.where(omask)[0]
    io = rng.choice(io, min(len(io), 4000), replace=False)
    Xt = np.concatenate([WCall[d][it] for d in dr]).astype(np.float64); yt = np.tile(y[it], ndr)
    Xo = np.concatenate([WCall[d][io] for d in dr]).astype(np.float64)
    Xt1 = np.c_[Xt, np.ones(len(Xt))]; Xo1 = np.c_[Xo, np.ones(len(Xo))]
    sc = Xt.std(0).mean()**2
    A = Xt1.T@Xt1 + lam_o*len(Xt)/len(Xo)*Xo1.T@Xo1 + alpha*len(Xt)*sc*np.diag(np.r_[np.ones(192),0])
    sol = np.linalg.solve(A, Xt1.T@yt)
    return sol[:192], sol[192]
def rhat_all(w, b): return np.einsum('dnk,k->dn', WCall, w.astype(np.float32)) + b
code = A_.code.values; stay = A_.stay.values
tm7 = code==7; off = ~tm7
y_ne = np.nan_to_num(RRall.rne1.values.astype(float))
res = (RRall.obs - RRall.pm).values.astype(float)
gk = GroupKFold(5)
for alpha in (1e-4, 1e-3, 1e-2):
    pr = np.zeros(len(y_ne))
    for tr, te in gk.split(np.zeros(len(y_ne)), y_ne, stay):
        msk_tr = np.zeros(len(y_ne), bool); msk_tr[tr] = True
        w, b = fit_lin(tm7 & msk_tr, y_ne, off & msk_tr, alpha=alpha)
        pr[te] = (WCm[te] @ w + b)
    yy = y_ne[tm7]; pp = pr[tm7]
    r2 = 1 - ((yy-pp)**2).sum()/((yy-yy.mean())**2).sum(); gain = np.cov(pp, yy)[0,1]/yy.var()
    pu = pp/gain
    ok = np.isfinite(res[tm7])
    aopt = np.cov(pu[ok], res[tm7][ok])[0,1]/pu[ok].var()
    print('alpha', alpha, 'CV R2 %.3f gain %.3f | resid-opt coef on unit readout %.2f' % (r2, gain, aopt))
for nm, ro, msk, yc in [('ins ROi5', ROi5, np.isin(code, [6,10]) & (A_.hi.values==1), 'dev1'), ('ne ROn4', ROn4, tm7, 'nedev1')]:
    pu = WCm[msk] @ np.asarray(ro['w'], float) + ro['w0']
    ok = np.isfinite(res[msk])
    print(nm, 'resid-opt coef %.2f' % (np.cov(pu[ok], res[msk][ok])[0,1]/pu[ok].var()), ' slope readout on target %.2f' % (np.cov(pu[ok], np.nan_to_num(A_[yc].values[msk][ok]))[0,1]/np.nan_to_num(A_[yc].values[msk][ok]).var()))
```