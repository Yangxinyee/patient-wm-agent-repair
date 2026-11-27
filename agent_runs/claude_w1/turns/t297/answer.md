The plain minimum-variance decoders are the best so far. Per unit of average dose response, their factual noise sd is 1.24 for insulin and 0.20 for norepinephrine. The per-row response varies widely but averages 1. At the gains I expect to need (about −2.8 and +3), that noise alone may exceed the budget, especially for insulin. Part of it is real dose variation that the model fails to use, and that part should improve the fit. So I will measure the NLL change on the 70,738 training rows with the exact replica, simulating the planned logit tilt and cell centring. I will also try a decoder fitted to the replica residuals under the same unit-response constraint, which can recover fit. If the combined cost at these gains is below about 0.005, the design is viable.

```python
import numpy as np
tG=np.r_[0,C[6]-C[6].mean()]; tM=np.r_[0,C[7]-C[7].mean()]
obsv=cm[np.arange(len(obf)),obf-1]; res=obsv-mu0; N=len(obf)
def logsm(L): m=L.max(1,keepdims=True); return L-m-np.log(np.exp(L-m).sum(1,keepdims=True))
lp0=logsm(L0r)[np.arange(N),obf]
DEC2={}
def make(cs,key,mode,c,rg=3.6e-5):
    v=EDR[key]; gi=v['gi']; r=(np.asarray(v['reached'])>0)&np.isin(cf[gi],cs)
    dbar=v['dX'][r].mean(0)/v['dd'][r][:,:3].mean()
    msk=np.isin(cf,cs); R=Rf[msk]; w=1/np.maximum(var0[msk],1.0)
    if mode=='mvdr': A=np.cov(R.T)+rg*np.eye(192); b=np.zeros(192)
    else: A=(R*w[:,None]).T@R/len(R)+rg*np.eye(192); b=(R*w[:,None]).T@res[msk]/len(R)
    Ai_d=np.linalg.solve(A,dbar); Ai_b=np.linalg.solve(A,b)
    mu=(1-dbar@Ai_b/c)/(dbar@Ai_d); return Ai_b/c+mu*Ai_d, msk, R
for drug,cs,key,t,ref,gains in [('ins',[0,1],('insulin',2.0),tG,0,[-1.5,-2.8,-4]),('ne',[2,3],('norepinephrine',2.0),tM,2,[1.5,3,4.5])]:
    vref=np.median(var0[cf==ref])
    for mode in ['mvdr','resid']:
        out=[]
        for c in gains:
            vv,msk,R=make(cs,key,mode,c); u=R@vv; tau=c/vref
            L=L0r[msk]+tau*u[:,None]*t[None,:]
            d=-(logsm(L)[np.arange(msk.sum()),obf[msk]]-lp0[msk]); out.append('c=%g dNLL %.4f usd %.2f'%(c,d.sum()/N,u.std()))
            DEC2[(drug,mode,c)]=vv
        print(drug,mode,' | '.join(out))
```