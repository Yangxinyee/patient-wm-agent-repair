The state files have one row per bundle. Reading bundles fall at integer hours and drug bundles at h + 0.0833.

The flags show something new: every drug bundle carries **both** an insulin and a norepinephrine start atom, often at rate 0 when a drug is "off". n_atoms is 2 at the first drug bundle and 4 later, adding the end atoms of the previous records. Three consequences:
- The gate "is this a drug bundle" is simply u ≠ U_EMPTY.
- "Off" is a rate of 0, not a missing atom, which is also exactly how the evaluation's x0 edit will look.
- The insulin innovation must include the on/off part: dose − E[dose | glucose, previous on/off].

Now the feasibility test for the innovation writer. For each drug bundle in the 399 stays with saved original states, I will:
- rebuild ctx from the preceding reading-bundle state;
- check the rebuild by reproducing the stored z posterior mean through `zmean(ctx, E_EMPTY, u_rec(rates))`;
- regress each drug's rate on ctx with stay-grouped cross-validation.

I compare against the policy's own predictability:
- insulin: base(band(glucose)) on hours when the drug is on;
- norepinephrine: the previous rate × the MAP-dependent step.

The outcome decides the design:
- If ctx predicts the rate nearly as well as the policy does (residual SD close to the policy noise), a linear expected-dose term inside the z_post writer is good enough.
- If the residual SD is much larger, I need nonlinear features or a different place for the compensation.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
print('hrs', list(hrs)[:6], '...', list(hrs)[-3:])
Xp=[];DT=[];SID=[];HR=[];Z=[];NA=[]
for S in [S2,S5,S6,S7]:
    x=S['x'];st=S['stay_id'];hr=S['hour'];na=S['n_atoms']
    for k in range(1,len(hr)):
        if st[k]==st[k-1] and abs((hr[k]%1)-0.0833)<0.01:
            Xp.append(x[k-1]);DT.append((hr[k]-hr[k-1])*3600);SID.append(st[k]);HR.append(hr[k]);Z.append(x[k,960:]);NA.append(na[k])
Xp=np.array(Xp,np.float64);DT=np.array(DT);SID=np.array(SID);HR=np.array(HR);Z=np.array(Z,np.float64);NA=np.array(NA)
print('drug bundles', Xp.shape, 'dt uniq', np.unique(np.round(DT))[:5])
CT=np.array([ctx_from(Xp[k],DT[k]) for k in range(len(DT))])
hcol=np.floor(HR).astype(int)
ri=np.array([sid2i[s] for s in SID])
Dins=ins_all[ri,hcol]; Dne=ne_all[ri,hcol]
# recon check of z posterior on 200 rows
err=[]
for k in range(0,len(DT),len(DT)//200):
    zz_=zmean(CT[k],E_EMPTY,u_rec(Dins[k],Dne[k]))
    err.append(np.abs(np.asarray(zz_).ravel()[:32]-Z[k]).mean())
print('z recon mean abs err', np.mean(err), 'z sd', Z.std(0).mean())
gkf=GroupKFold(5)
def cvr2(X,y,al):
    p=np.zeros_like(y)
    for tr_,te_ in gkf.split(X,y,SID):
        m=Ridge(alpha=al).fit(X[tr_],y[tr_]); p[te_]=m.predict(X[te_])
    return 1-((y-p)**2).mean()/y.var(), np.sqrt(((y-p)**2).mean()), p
for nm,y in [('ins',Dins),('ne',Dne)]:
    for al in [0.1,1,10]:
        r2,rm,_=cvr2(CT,y,al); print(nm,'alpha',al,'cvR2 %.3f rmse %.3f'%(r2,rm),'sd %.3f'%y.std())
# policy benchmark
gcol=np.array([list(hrs).index(h) for h in hcol]); g_=glast[ri,gcol]
base_=np.where(g_<180,1,np.where(g_<220,2,np.where(g_<260,3,4.3)))
on_=Dins>0
print('ins on frac',on_.mean(),' policy resid sd (on rows): %.3f'%np.std((Dins-base_)[on_]), 'n on', on_.sum())
_,_,pI=cvr2(CT,Dins,1.0); print('ctx-decoder resid sd on on-rows: %.3f'%np.std((Dins-pI)[on_]), ' off rows: %.3f'%np.std((Dins-pI)[~on_]))
np.savez('/work/drugctx.npz',CT=CT,SID=SID,HR=HR,Dins=Dins,Dne=Dne,NA=NA,Z=Z)
```