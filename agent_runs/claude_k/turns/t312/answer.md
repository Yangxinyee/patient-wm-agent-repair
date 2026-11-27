The last run showed my writer v3 would have been dead on arrival. The z token is tiny (its spread across dimensions is only 0.18), so the gain I computed actually shrinks z_in's columns. Earlier runs had already put a 2.6-unit normalised shift on that token for insulin, and 16 units for norepinephrine, and still changed h by only about 2% of its natural variation. The prelude simply barely attends to the z token, and pushing harder saturates its layer norm. So the z route is closed and I will not spend a forward pass on it.

Every way bundle information reaches the state goes through attention. Insulin's rate is carried poorly while norepinephrine's is carried strongly (the model needs the previous norepinephrine rate to predict titration). The realistic plan is therefore a reader in the value head on the model's own dose pathway: gated, paired units on q that act only on MAP codes and/or glucose codes.

This run:
- reverts the writer;
- rebuilds q at reading bundles, using the documented view formula, from the stored emulation states (xo = original, xe = model shown doubled doses);
- regresses the change in q on lag-1 and lag-2 rates;
- computes each drug's efficiency η = bᵀΣ⁻¹b against the natural q covariance, which sets the factual cost of reaching a given counterfactual effect;
- inspects the factual output file so I can measure how a reader correlates with the model's errors.

What decides the plan:
- If η for norepinephrine is ≫1 per unit², a norepinephrine reader is nearly free in factual cost.
- If insulin's η is ≪1, an insulin reader can only partially close the gap within the 0.005 budget.

```python
import numpy as np, pandas as pd, json, math
from scipy.special import erf
for n in ['z_in.bias','z_in.weight','z_post.0.bias','z_post.0.weight','z_post.2.bias','z_post.2.weight']: drop_edit(n)
print('edited:', edited())
def A(n): return np.array(np.asarray(load_original(n)), dtype=np.float64)
def LN(x,w,b):
    mu=x.mean(-1,keepdims=True); v=((x-mu)**2).mean(-1,keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*w+b
def gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))
fa=A('core.flow.a'); onw,onb=A('core.out_norm.weight'),A('core.out_norm.bias')
g0w,g0b,g2w,g2b=A('gap.0.weight'),A('gap.0.bias'),A('gap.2.weight'),A('gap.2.bias')
p0w,p0b,p2w,p2b=A('z_prior.0.weight'),A('z_prior.0.bias'),A('z_prior.2.weight'),A('z_prior.2.bias')
zhw,zhb=A('z_head.weight'),A('z_head.bias'); qnw,qnb=A('query_norm.weight'),A('query_norm.bias'); te1=A('type_emb.weight')[1]
def view_q(X, dts):
    out=X[:,:192]; h=X[:,192:960].reshape(-1,4,192)
    hf=h*np.exp(-dts[:,None,None]*np.exp(fa)[None,None,:])
    mh=LN(hf.mean(1),onw,onb)
    g=gelu((np.log(dts)/10)[:,None]@g0w.T+g0b)@g2w.T+g2b
    ctx=out+mh+g
    zp=gelu(ctx@p0w.T+p0b)@p2w.T+p2b
    return LN(ctx+zp[:,:32]@zhw.T+zhb+te1,qnw,qnb)
hour=np.asarray(hr,float); st=np.asarray(sid); rI=np.asarray(rateI,float); rN=np.asarray(rateN,float)
isread=np.abs(hour-np.round(hour))<0.01
print('rows',len(hour),'read',isread.sum(),'| rateI>0 at drug rows',(rI[~isread]>0).mean().round(3),'rateI>0 at read rows',(rI[isread]>0).mean())
# lag rates by stay & floor hour of drug bundles
dmap={}
for i in np.where(~isread)[0]: dmap[(st[i],int(np.floor(hour[i]+1e-6)))]=(rI[i],rN[i])
k=np.where(isread[1:]&(st[1:]==st[:-1]))[0]+1
dts=(hour[k]-hour[k-1])*3600.0
def lag(i,L,j): return dmap.get((st[i],int(round(hour[i]))-L),(0.0,0.0))[j]
F=np.array([[lag(i,1,0),lag(i,2,0),lag(i,1,1),lag(i,2,1)] for i in k])
print('n read rows',len(k),'dt pct',np.percentile(dts,[0,5,50,95,100]).round(0),'| F means',F.mean(0).round(3),'| check vs lI1 corr',np.corrcoef(F[:,0],np.asarray(lI1,float)[k])[0,1].round(3))
Qo=view_q(np.asarray(xo,float)[k-1],dts); Qe=view_q(np.asarray(xe,float)[k-1],dts); dQ=Qe-Qo
b=np.linalg.lstsq(np.c_[F,np.ones(len(k))],dQ,rcond=None)[0]; R=dQ-np.c_[F,np.ones(len(k))]@b
Sn=np.cov(Qo.T); Sr=np.cov(R.T)
def eta(bv,S,lam=1e-2):
    Si=np.linalg.inv(S+lam*np.trace(S)/192*np.eye(192)); return float(bv@Si@bv), Si@bv
print('|dQ| rms %.3f, |q-mean| rms %.3f'%(np.sqrt((dQ**2).sum(1).mean()), np.sqrt(((Qo-Qo.mean(0))**2).sum(1).mean())))
for i,nm in enumerate(['I1','I2','N1','N2']):
    print(nm,'|b| %.3f eta_nat %.3f eta_res %.3f'%(np.linalg.norm(b[i]),eta(b[i],Sn)[0],eta(b[i],Sr)[0]))
QREAD=dict(k=k,dts=dts,F=F,Qo=Qo,Qe=Qe,b=b,Sn=Sn,Sr=Sr)
fd=pd.read_csv('/work/fwd_4/factual_original.csv'); print(fd.columns.tolist(), len(fd)); print(fd.head(3).to_string())
```