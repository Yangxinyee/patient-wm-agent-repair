The scan is invalid because of a bug in my solver. It multiplies the ridge penalty by the number of rows, which makes it about 19,000 times stronger than the sklearn alpha that gave R² 0.61. That is why even λ = 0 only reaches R² 0.19.

The mismatch is itself telling. The best readout needs almost no ridge, so the dose information sits in input directions with very little variance on target rows. Those same directions vary a lot on other rows, which explains the explosion off target.

Next I will rerun the λ scan with the penalty on sklearn's absolute scale. The trade-off I hope for: at some λ, R² stays near 0.5 while the off-target range drops to about ±20.

```python
def fit_ro5(tmask,omask,ycol,Pp,lam,alpha=1e-3,ndr=2):
    it=np.where(tmask)[0]; io=np.where(omask)[0]; y=np.nan_to_num(A_[ycol].values[it])
    Xt=np.concatenate([WCall[d][it] for d in range(ndr)],0).astype(np.float64)@Pp
    Xo=np.concatenate([WCall[d][io] for d in range(ndr)],0).astype(np.float64)@Pp
    yt=np.tile(y,ndr); gt=np.tile(stayA[it],ndr); go=np.tile(stayA[io],ndr)
    wo=lam*len(yt)/len(Xo)
    def solve(Xt_,yt_,Xo_):
        mt=Xt_.mean(0); my=yt_.mean(); Xc=Xt_-mt; Xoc=Xo_-mt
        Am=Xc.T@Xc+wo*(Xoc.T@Xoc)+alpha*np.eye(192); bm=Xc.T@(yt_-my)-wo*Xoc.T@np.full(len(Xo_),my)
        w=np.linalg.solve(Am,bm); return w, my-mt@w
    ust=np.unique(stayA[it]); rng=np.random.default_rng(0); fold=dict(zip(ust,rng.integers(0,5,len(ust))))
    ft=np.array([fold[s] for s in gt]); fo=np.array([fold.get(s,-1) for s in go]); fo[fo<0]=rng.integers(0,5,(fo<0).sum())
    pr=np.zeros(len(yt))
    for k in range(5):
        w_,b_=solve(Xt[ft!=k],yt[ft!=k],Xo[fo!=k]); pr[ft==k]=Xt[ft==k]@w_+b_
    r2=1-np.var(yt-pr)/np.var(yt); gain=np.cov(pr,yt)[0,1]/np.var(yt,ddof=1)
    w_,b_=solve(Xt,yt,Xo); w=(Pp@w_)/gain; w0=b_/gain
    R=np.einsum('dnk,k->dn',WCall,w.astype(np.float32))+w0
    return dict(w=w,w0=w0,gain=gain,r2=r2),R
res_l={}
for lam in [0.0,1e-4,1e-3,1e-2,1e-1]:
    ro_,R_=fit_ro5(ins_t,offi,'dev1',Pp_i,lam); res_l[lam]=(ro_,R_)
    print('lam %g: r2 %.3f gain %.3f | target pct0.05/99.95 %s | glu&~hi %s | MAP %s'%(lam,ro_['r2'],ro_['gain'],
          np.round(np.percentile(R_[:,ins_t],[0.05,99.95]),1).tolist(),np.round(np.percentile(R_[:,glu&~hiA],[0.05,99.95]),1).tolist(),np.round(np.percentile(R_[:,mapc],[0.05,99.95]),1).tolist()))
```