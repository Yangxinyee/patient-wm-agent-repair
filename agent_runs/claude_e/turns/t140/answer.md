Partial success.

**Insulin readout.** With weights orthogonal to the code embeddings it keeps R² 0.61 and gain 0.61 on its target rows. But it is still huge where it should be silent: −62 to +133 on glucose rows with no insulin, −86 to +125 on MAP rows. So the code embedding was not the main cause. The readout extrapolates along state directions that differ between insulin and non-insulin bundles, which never vary within its training rows.

**Norepinephrine readout.** It is well-behaved off target (|r| ≤ 3.1).

Gating ±130 with linear gates would need pre-activations above 1000. That is risky if the GPU runs matrix products at reduced precision.

So I will shrink the insulin readout off target directly. I fit a generalised ridge: squared error on the target rows plus λ times the readout's squared value on non-target rows (glucose rows without insulin, and MAP rows). I scan λ and report the trade-off between R² and gain on target rows and the off-target range.

I hope a small λ keeps R² near 0.55 while cutting the off-target range to roughly ±20. Then moderate gates (scale about 25–40) suffice.

```python
def fit_ro5(tmask,omask,ycol,Pp,lam,alpha=1e-3,ndr=2):
    it=np.where(tmask)[0]; io=np.where(omask)[0]; y=np.nan_to_num(A_[ycol].values[it])
    Xt=np.concatenate([WCall[d][it] for d in range(ndr)],0).astype(np.float64)@Pp
    Xo=np.concatenate([WCall[d][io] for d in range(ndr)],0).astype(np.float64)@Pp
    yt=np.tile(y,ndr); gt=np.tile(stayA[it],ndr); go=np.tile(stayA[io],ndr)
    wo=lam*len(yt)/len(Xo)
    def solve(Xt_,yt_,Xo_):
        mt=Xt_.mean(0); my=yt_.mean(); Xc=Xt_-mt; Xoc=Xo_-mt
        Am=Xc.T@Xc+wo*(Xoc.T@Xoc)+alpha*len(yt_)*np.eye(192); bm=Xc.T@(yt_-my)-wo*Xoc.T@(np.full(len(Xo_),my))
        w=np.linalg.solve(Am,bm); return w, my-mt@w
    pr=np.zeros(len(yt)); ust=np.unique(stayA[it]); rng=np.random.default_rng(0); fold=dict(zip(ust,rng.integers(0,5,len(ust))))
    ft=np.array([fold[s] for s in gt]); fo=np.array([fold.get(s,rng.integers(0,5)) for s in go])
    for k in range(5):
        w_,b_=solve(Xt[ft!=k],yt[ft!=k],Xo[fo!=k]); pr[ft==k]=Xt[ft==k]@w_+b_
    r2=1-np.var(yt-pr)/np.var(yt); gain=np.cov(pr,yt)[0,1]/np.var(yt,ddof=1)
    w_,b_=solve(Xt,yt,Xo); w=(Pp@w_)/gain; w0=b_/gain
    R=np.einsum('dnk,k->dn',WCall,w.astype(np.float32))+w0
    return dict(w=w,w0=w0,gain=gain,r2=r2),R
Pp_i=perp(6,[10,7,5]); offi=(glu&~hiA)|mapc
for lam in [0.0,1e-3,1e-2,3e-2,1e-1]:
    ro_,R_=fit_ro5(ins_t,offi,'dev1',Pp_i,lam)
    print('lam %g: r2 %.3f gain %.3f | target pct0.05/99.95 %s | glu&~hi %s | MAP %s'%(lam,ro_['r2'],ro_['gain'],
          np.round(np.percentile(R_[:,ins_t],[0.05,99.95]),1).tolist(),np.round(np.percentile(R_[:,glu&~hiA],[0.05,99.95]),1).tolist(),np.round(np.percentile(R_[:,mapc],[0.05,99.95]),1).tolist()))
```