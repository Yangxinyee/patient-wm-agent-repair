The regression gives a mixed answer.
- **Insulin works as planned.** The model's glucose error regresses strongly on the context-based insulin residual: −1.49 (se 0.37) at lag 1 and −1.79 (0.38) at lag 2. The expected-dose terms are about 0, apart from one marginal lag-3 term. So the model has absorbed the policy and is missing roughly the true direct effect. An insulin residual injection should improve NLL, not cost it.
- **Norepinephrine residual is mostly leak.** Its lag-1 coefficient on MAP is only 0.17 (0.81). That fits a residual made mostly of leaked policy variation: the true residual has sd 0.073 against 0.254 for this one, so dilution predicts about 3 × 0.0053/0.065 ≈ 0.25. A norepinephrine correction will therefore cost some NLL. A rough Gaussian estimate says a lag-1 gain of about 2–3 per mcg/kg/min costs only about 0.001 nats per reading on average, which is affordable, but I'll measure it exactly.
- **Error sds:** finger 7.5, lab 14.7, arterial 4.9, cuff 12.8.

Next I'll find the token-space injection directions. The z token is set to an exact copy of u, and I take finite-difference Jacobians of the next reading's predicted means (finger, lab, arterial, cuff) with respect to a perturbation of that token, at random drug bundles. Ridge least squares then gives a direction that moves both glucose codes by −1 with MAP unchanged, and one that moves both MAP codes by +1 with glucose unchanged. I'll also check two things:
- the copy itself leaves the original predictions unchanged;
- the percentile errors of a single linear dose-from-u decoder, using one "any drug" gate.

What would refute the plan:
- a large spread of the effect across bundles relative to its mean;
- a required token norm that is large compared with u.

```python
import numpy as np, torch, time
t0=time.time()
ALLC=[CODE_ID['glucose_finger'],CODE_ID['glucose_lab'],CODE_ID['map_arterial'],CODE_ID['map_cuff']]
ue=u_empty.numpy().astype(np.float64)
Ud=np.vstack([U,UE]); yi=np.r_[np.where(hi,ri,0),TE[:,0]*(TE[:,2]>0)]; yn=np.r_[np.where(hn,rn,0),TE[:,1]*(TE[:,3]>0)]
fac_i=np.r_[np.ones(len(U)),np.tile([0,.5,2,3,1,1,1,1],len(UE)//8)]; fac_n=np.r_[np.ones(len(U)),np.tile([1,1,1,1,0,.5,2,3],len(UE)//8)]
Xd=np.c_[Ud-ue,np.ones(len(Ud))]; DEC={}
for nm,y,fc in [('ins',yi,fac_i),('ne',yn,fac_n)]:
    a=np.linalg.lstsq(np.vstack([Xd,np.r_[np.zeros(192),1.0][None]]),np.r_[y,0.0],rcond=None)[0]; er=Xd@a-y; DEC[nm]=a
    print(nm,'dec |err| p50/p90/p99 by f',[(f,np.round(np.percentile(np.abs(er[fc==f]),[50,90,99]),2).tolist()) for f in [0,.5,1,2,3]])
def regp(reg,c,tn,Bt):
    if c in reg: tl,zl,vl=reg[c]; return torch.ones(Bt,dtype=T),torch.full((Bt,),(tn-tl)*3600.0,dtype=T),torch.full((Bt,),float(zl),dtype=T),torch.full((Bt,),int(vl),dtype=torch.long)
    return torch.zeros(Bt,dtype=T),torch.zeros(Bt,dtype=T),torch.zeros(Bt,dtype=T),torch.zeros(Bt,dtype=torch.long)
def nextmeans(W,o2,h2,dtn,reg,tn):
    Bt=o2.shape[0]; views=bview(W,o2,h2,dtn,zs=torch.zeros(Bt,1,32,dtype=T)); M_=[]
    for c in ALLC:
        p=bvalue_probs(W,views,c,*regp(reg,c,tn,Bt)); M_.append((p[:,1:]*torch.tensor(CENd[c],dtype=T)).sum(1))
    return torch.stack(M_,1)
EPS=0.05; DM=torch.cat([torch.zeros(1,192,dtype=T),EPS*torch.eye(192,dtype=T),-EPS*torch.eye(192,dtype=T)],0)
def jac_stay(W,B_,s,rng,nb=3):
    times=[b[0] for b in B_]
    elig=[k for k in range(1,len(B_)-1) if any(a['act'] and a['kind']==0 for a in B_[k][1]) and any(a['code'] in MEASC and a['kind']==0 for a in B_[k+1][1])]
    pick=set(rng.choice(elig,min(nb,len(elig)),replace=False).tolist()) if elig else set()
    out=torch.tensor(X0row[:192],dtype=T)[None]; h=torch.tensor(X0row[192:960],dtype=T).reshape(1,4,192)
    ea,ua=split_fix(B_[0][1]); Ms=[set_enc(W,ea,'encode')]; ts=[times[0]]; reg={}
    for a in ea:
        if a['code'] in MEASC and a['kind']==0: reg[a['code']]=(times[0],a['num'][0],a['vb'])
    J=[]
    for k in range(1,len(B_)):
        t=times[k]; dt=(t-times[k-1])*3600.0; ea,ua=split_fix(B_[k][1])
        for a in ea:
            if a['code'] in MEASC and a['kind']==0: reg[a['code']]=(t,a['num'][0],a['vb'])
        e=set_enc(W,ea,'encode'); u=set_enc(W,ua,'encode_action') if ua else u_empty
        Ms.append(e); ts.append(t); Mt=torch.stack(Ms)[None]; ages=torch.tensor([(t-tj)*3600.0 for tj in ts],dtype=T)
        if k in pick:
            tn=times[k+1]; dtn=(tn-t)*3600.0; Bt=DM.shape[0]
            o2,h2,_=bstep_z(W,out.repeat(Bt,1),h.repeat(Bt,1,1),dt,e[None].repeat(Bt,1),u[None].repeat(Bt,1),Mt.repeat(Bt,1,1),ages,zfn=lambda c_,e_,u_,i_: u_+DM)
            Mm=nextmeans(W,o2,h2,dtn,reg,tn)
            o0,h0,_=bstep(W,out,h,dt,e[None],u[None],Mt,ages); M0=nextmeans(W,o0,h0,dtn,reg,tn)
            J.append(dict(s=s,k=k,J=((Mm[1:193]-Mm[193:])/(2*EPS)).T.numpy(),copydiff=float((Mm[0]-M0[0]).abs().max())))
        out,h,_=bstep(W,out,h,dt,e[None],u[None],Mt,ages)
    return J
rng=np.random.default_rng(11); JJ=[]
for s,B_ in TRB.items():
    JJ+=jac_stay(Wb,B_,s,rng)
    if time.time()-t0>480: break
Js=np.stack([j['J'] for j in JJ]); print('bundles',len(JJ),'copy-vs-orig max diff',round(max(j['copydiff'] for j in JJ),4),'t',round(time.time()-t0))
print('u token norm (centered) median',round(np.median(np.linalg.norm(U-U.mean(1,keepdims=True),axis=1)),2))
DIRS={}
for nm,Tg in [('ins',np.array([-1,-1,0,0.])),('ne',np.array([0,0,1,1.]))]:
    A=np.einsum('bci,bcj->ij',Js,Js); bv=np.einsum('bci,c->i',Js,Tg)
    for lam in [1e-3,1e-2,1e-1,1]:
        d=np.linalg.solve(A+lam*len(Js)*np.eye(192),bv); eff=np.einsum('bci,i->bc',Js,d); DIRS[(nm,lam)]=d
        print(nm,'lam',lam,'|d|',round(np.linalg.norm(d),3),'eff mean',np.round(eff.mean(0),2),'sd',np.round(eff.std(0),2))
np.savez('/work/jac.npz',Js=Js,sk=np.array([(j['s'],j['k']) for j in JJ]))
```