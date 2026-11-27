Three results:
- **Cheap units exist.** Prelude FFN units 721, 126 and 423 each contribute about 0.015, against a median of 0.26, so replacing them should cost almost nothing.
- **`u_rec` is exact.** It reproduces the true drug-bundle u with zero error (absent drug = rate 0), so I can encode edited doses exactly.
- **Context decoders are available.** The expected-rate decoders (`wA`, `wB` on ctx) are in the decoders file.

Next I'll build the training set for the new units. For every drug bundle I recompute the prelude FFN inputs under dose multipliers (insulin ×0 to ×3 with NE as recorded, and vice versa). I'll then fit three linear read-outs on those inputs: insulin innovation (scaled rate minus context-expected rate), NE innovation, and a drug-vs-reading gate. The innovation read-outs should reach high cross-validated R² (≳0.9) and the gate should separate drug from reading bundles with a wide margin. If either fails, the FFN-unit route won't be clean enough.

```python
ctxs=np.array([np.asarray(ctx_from(xo[k-1],(hr[k]-hr[k-1])*3600),float).reshape(-1)[:192] for k in kd])
EA=ctxs@dz_['wA']+float(dz_['bA']); EBn=ctxs@dz_['wB']+float(dz_['bB'])
ri0=np.nan_to_num(RI); rn0=np.nan_to_num(RN)
print('ctx-dec corr ins %.3f ne %.3f'%(np.corrcoef(EA,ri0)[0,1],np.corrcoef(EBn,rn0)[0,1]))
VAR=[(m,1.0) for m in [0,.5,1,1.5,2,3]]+[(1.0,m) for m in [0,.5,1.5,2,3]]
nd=len(kd); FA=np.zeros((len(VAR),nd,4,192),np.float32)
base=[]
for i,k in enumerate(kd):
    e,_u=[np.asarray(x,float) for x in enc_e_var((int(sid[k]),round(float(hr[k]),3)),'none')]
    hp=xo[k-1,192:960].reshape(4,192)*np.exp(-(hr[k]-hr[k-1])*3600*np.exp(fa)); zt=Zw@xo[k,960:992]+Zb
    for v,(mA,mB) in enumerate(VAR):
        u=np.asarray(u_rec(float(ri0[i]*mA),float(rn0[i]*mB))).reshape(-1)[:192]
        FA[v,i]=prelude(hp,e,u,zt)[0]
mAv=np.array([m for m,_ in VAR]); mBv=np.array([m for _,m in VAR])
TA=(mAv[:,None]*ri0[None]-EA[None]); TB=(mBv[:,None]*rn0[None]-EBn[None])
stay_fold=np.array([hash(int(sid[k]))%5 for k in kd])
def ridge_cv(X,y,fold,al=1e-2):
    pred=np.zeros_like(y)
    for f in range(5):
        tr=fold!=f; Xt=np.c_[X[tr],np.ones(tr.sum())]; w=np.linalg.solve(Xt.T@Xt+al*np.eye(Xt.shape[1]),Xt.T@y[tr]); pred[~tr]=np.c_[X[~tr],np.ones((~tr).sum())]@w
    Xt=np.c_[X,np.ones(len(X))]; w=np.linalg.solve(Xt.T@Xt+al*np.eye(Xt.shape[1]),Xt.T@y); return w,pred
Xd=FA.reshape(-1,192).astype(float); fd=np.repeat(np.tile(stay_fold,len(VAR))[:,None],4,1).ravel() if False else np.tile(np.repeat(stay_fold,4),len(VAR))
yA=np.repeat(TA.reshape(-1),4); yB=np.repeat(TB.reshape(-1),4)
wAf,pA_=ridge_cv(Xd,yA,fd); wBf,pB_=ridge_cv(Xd,yB,fd)
for nm,y,p in [('A',yA,pA_),('B',yB,pB_)]: print(nm,'sd %.3f cvR2 %.4f resid sd %.3f'%(y.std(),1-((y-p)**2).mean()/y.var(),(y-p).std()))
Xr=FEAT[typ==0].reshape(-1,192).astype(float)
Xg=np.r_[Xd,Xr]; yg=np.r_[np.ones(len(Xd)),-np.ones(len(Xr))]; fg=np.r_[fd,np.repeat(np.array([hash(int(sid[k]))%5 for k in np.where(typ==0)[0]]),4)]
wGf,pG_=ridge_cv(Xg,yg,fg); print('gate drug min %.3f p01 %.3f | reading max %.3f p99 %.3f'%(pG_[:len(Xd)].min(),np.percentile(pG_[:len(Xd)],1),pG_[len(Xd):].max(),np.percentile(pG_[len(Xd):],99)))
```