Results from the last run:
- **Write route through the z token is not usable.** Gains vary from zero to large across bundles, and the response saturates: at injection 0.5 the gain is already only about 0.3 of the small-step derivative. I drop it.
- **Spare capacity on the read side:** z_prior has 125 dead hidden units.

**New candidate: read-out only.** Dead z_prior units read the dose signal the state already carries, and z_head maps it to a prediction shift. This only works if the context change caused by a dose change lies in a direction the natural context rarely moves along. Then a read-out with gain 1 for a dose change would add little noise to factual predictions.

**This run:**
- Compute the one-step context change when the insulin dose is doubled, per unit dose, on the insulin bundles.
- Fit a constrained regression (dose deviation ~ context, with unit gain on that change) and report the out-of-sample noise in dose units.
- Check how the response persists two bundles later and how the read-out behaves on all other bundles.

**Verdict:** factual noise well under about 0.5 dose units, with a stable gain, makes the read-out viable. Noise near the size of the deviation itself rules it out.

```python
import numpy as np, torch, time
t0=time.time()
def step_ku(k, u, hp=None, op=None):
    if hp is None: hp = X3t[k-1,192:960].reshape(4,192); op = X3t[k-1,:192]
    dtk = dts[k]; hf = hp*torch.exp(-dtk*torch.exp(fa)); ctx = ctx_from(op, hf, dtk)
    z = zpost_mean(ctx, Ee[k], u)
    idx_m = np.arange(max(j0[k], k+1-128), k+1); ages = (tmin[k]-tmin[idx_m])*60.0
    return core_step(hf, Ee[k], u, z, Ee[idx_m], ages, Vd)
kiA = np.asarray(ki).astype(int); devA = np.asarray(dev, float)
rI = np.array([RI.get((int(sid[k]), int(INFO[k]['t']//60)), np.nan) for k in kiA])
okI = np.array([ok[k] and k+3<N and sid[k+3]==sid[k] for k in kiA]) & np.isfinite(rI) & (rI>0)
print('ins bundles', len(kiA), 'usable', okI.sum(), 'rate mean', np.nanmean(rI).round(2))
Xf1=[]; Dl=[[],[],[]]; idxs=[]
for i in np.where(okI)[0]:
    k=int(kiA[i]); outs=[]
    for u in (Uu[k], u_for(k,2.0,1.0)):
        h,o = step_ku(k,u); cs=[ctx_after(k,h,o)]
        for kk in (k+1,k+2):
            h,o = step_ku(kk, Uu[kk], h, o); cs.append(ctx_after(kk,h,o))
        outs.append(cs)
    Xf1.append(outs[0][0].numpy()); idxs.append(i)
    for L in range(3): Dl[L].append(((outs[1][L]-outs[0][L])/rI[i]).numpy())
Xf1=np.array(Xf1); Dl=[np.array(x) for x in Dl]; idxs=np.array(idxs); yv=devA[idxs]
mu=Xf1.mean(0); Xc=Xf1-mu; db=Dl[0].mean(0)
rng=np.random.default_rng(1); perm=rng.permutation(len(yv)); A_,B_=perm[::2],perm[1::2]
def fitw(ix, lam):
    S=np.cov(Xc[ix].T)+lam*np.eye(192); c=Xc[ix].T@(yv[ix]-yv[ix].mean())/len(ix)
    Si=np.linalg.inv(S); w0=Si@c; lamc=(1-db@w0)/(db@Si@db); return w0+lamc*(Si@db)
for lam in [1e-4,1e-3,1e-2]:
    w=fitw(A_,lam); r=Xc[B_]@w-(yv[B_]-yv[A_].mean())
    g=Dl[0][B_]@w
    print(f'lam {lam}: OOS resid sd {r.std():.3f} (dev sd {yv.std():.3f}), gain lag1 mean {g.mean():.2f} sd {g.std():.2f}, lag2 {(Dl[1][B_]@w).mean():.2f}, lag3 {(Dl[2][B_]@w).mean():.2f}')
w=fitw(np.arange(len(yv)),1e-3); wI_=w; muI_=mu
CA=np.stack([ctx_nat(int(k)).numpy() for k in np.where(ok)[0]])
sa=(CA-mu)@w; print('readout over all bundles: mean', sa.mean().round(3), 'sd', sa.std().round(3), 'p5/p95', np.percentile(sa,[5,95]).round(2), 'time', round(time.time()-t0,1))
```