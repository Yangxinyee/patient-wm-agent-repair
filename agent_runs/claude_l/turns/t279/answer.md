The interfaces are now clear. `rollout(W, …)` and `tf_stay(W, …)` both run the standard core step. That step computes the z token as z_in(z_post([ctx, e, u])). So a weight-level edit of z_post and z_in automatically appears in both the teacher-forced fit and the dose rollouts. Here ctx is the context vector the core builds from the state before each bundle.

The plan, with no memory loop since decoding from the state was too noisy:
- **Copy u.** The z token becomes a PCA copy of u. Prelude heads then split 50/50 between two identical tokens, so the original computation is unchanged.
- **Add the residual.** At drug bundles, add g·r·w, where r = decoded dose − expected dose. The expected dose comes from ctx, so the policy-predictable part stays absorbed and the factual fit should not suffer.

This run collects teacher-forced data on 320 training stays and checks four things:
1. Can the expected dose be decoded linearly from ctx about as well as the policy GBM (resid sd ≈ 0.35 insulin, 0.073 NE)?
2. Can the dose be decoded from u, including for edited doses (x0, x1/2, x2, x3)?
3. Can 30 PCA components reproduce u?
4. Does the architecture sample z in the core? That would add noise once the z token is attended.

```python
import time, numpy as np, pandas as pd, torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict as cvp
t0=time.time()
Wb=dict(W0); Wb['z_head.weight']=torch.zeros_like(W0['z_head.weight'])
scol='stay_id' if 'stay_id' in tu.columns else [c for c in tu.columns if 'stay' in c][0]
s0=list(BUN.keys())[0]; AE=None
for ae in [('hospital_admission','icu_admission'),('icu_admission',),('hospital_admission',),()]:
    B2=build_bundles(tu[tu[scol]==s0],ae)
    if len(B2)==len(BUN[s0]) and all(abs(a[0]-b[0])<1e-9 and len(a[1])==len(b[1]) for a,b in zip(B2,BUN[s0])): AE=ae; break
print('AE',AE)
tr=pd.read_csv('/data/train.csv')
ids=tr[scol].unique(); rng=np.random.default_rng(7); sel=set(rng.choice(ids,320,replace=False).tolist())
TRB={}; DD=[]; RR=[]
for s,ds in tr[tr[scol].isin(sel)].groupby(scol, sort=False):
    B_=build_bundles(ds,AE); TRB[s]=B_
    R,D=tf_stay(Wb,B_)
    for d in D: d['s']=s
    DD+=D; RR+=[(s,r['k'],r['code'],r['nll']) for r in R]
    if time.time()-t0>450: break
print('stays',len(TRB),'drug',len(DD),'reads',len(RR),'t',round(time.time()-t0))
C=np.stack([d['ctx'] for d in DD]).astype(np.float64); U=np.stack([d['u'] for d in DD]).astype(np.float64)
ri=np.array([d['ins'] for d in DD]); rn=np.array([d['ne'] for d in DD])
hi=np.array([d['hi'] for d in DD],bool); hn=np.array([d['hn'] for d in DD],bool)
ue=u_empty.numpy().astype(np.float64)
def r2(y,p): return 1-np.var(y-p)/np.var(y)
for nm,m,y in [('ins',hi,ri),('ne',hn,rn)]:
    rc=RidgeCV(alphas=[1e-4,1e-3,1e-2,1e-1,1,10]).fit(C[m],y[m]); p=cvp(RidgeCV(alphas=[rc.alpha_]),C[m],y[m],cv=5)
    print(nm,'n',m.sum(),'pihat(ctx) R2',round(r2(y[m],p),3),'resid sd',round(np.std(y[m]-p),3),'alpha',rc.alpha_)
W0c[0]=Wb
idx=rng.choice(len(DD),min(700,len(DD)),replace=False); UE=[]; TE=[]
for j in idx:
    d=DD[j]; atoms=TRB[d['s']][d['k']][1]
    for code,f in [(8,0.0),(8,0.5),(8,2.0),(8,3.0),(9,0.0),(9,0.5),(9,2.0),(9,3.0)]:
        UE.append(u_for(atoms,d['t'],(code,f),-1.0).numpy()); TE.append((d['ins']*(f if code==8 else 1),d['ne']*(f if code==9 else 1),d['hi'],d['hn']))
UE=np.array(UE,dtype=np.float64); TE=np.array(TE,dtype=np.float64)
for nm,col,hc in [('ins',0,2),('ne',1,3)]:
    m=TE[:,hc]>0; y=TE[m,col]; Xa=np.vstack([U[(hi if col==0 else hn)],UE[m]]); ya=np.r_[(ri if col==0 else rn)[(hi if col==0 else hn)],y]
    lr=RidgeCV(alphas=[1e-6,1e-4,1e-2]).fit(Xa,ya); pe=lr.predict(UE[m])
    print(nm,'dose-from-u fit R2',round(r2(ya,lr.predict(Xa)),4),'edited max err',round(np.abs(pe-y).max(),3),'mean err by f>1',round(np.mean((pe-y)[y>(ri.max() if col==0 else rn.max())]),3) if (y>(ri.max() if col==0 else rn.max())).any() else None)
Uall=np.vstack([U,UE,ue[None]]); mu=Uall.mean(0); S=np.linalg.svd(Uall-mu,full_matrices=False)
ev=S[1]**2/np.sum(S[1]**2)
for K in [10,20,28,30]:
    Pk=S[2][:K]; rec=mu+(Uall-mu)@Pk.T@Pk; rel=np.linalg.norm(rec-Uall,axis=1)/np.linalg.norm(Uall-Uall.mean(1,keepdims=True),axis=1)
    print('PCA K',K,'cum var',round(ev[:K].sum(),6),'rel err med/max',round(np.median(rel),5),round(rel.max(),5))
np.savez('/work/s1data.npz',C=C,U=U,ri=ri,rn=rn,hi=hi,hn=hn,UE=UE,TE=TE)
txt=open('/model/ARCHITECTURE.md').read().splitlines()
print([l[:160] for l in txt if ('posterior' in l.lower() or 'z_post' in l or 'sampl' in l.lower())][:8])
print('time',round(time.time()-t0))
```