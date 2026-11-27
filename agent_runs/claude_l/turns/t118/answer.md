The CSV has observed value, predicted mean and NLL per reading, so I can compute the model's error directly.

I'll regress the error, for each glucose and MAP code, on the policy-predicted dose and the dose residual at lags 1–3 (insulin for glucose, norepinephrine for MAP). The data-based conditional effects I expect for the residual are about −2.0 (insulin, lag 1) and +3.2 (norepinephrine, lag 1).
- **Absorbed policy:** the residual coefficients sit far from zero and the predicted-dose coefficients near zero.
- **Not absorbed:** both carry similar coefficients.

```python
import numpy as np
sidx_tu={s:i for i,s in enumerate(sid_tu)}
Ptu={d: pol[d].predict(Xtu.reshape(-1,Xtu.shape[2])).reshape(len(sid_tu),25) for d in DRUGS}
def dosemat(d, si, t, lag):
    tt=t-lag; ok=(tt>=0)
    dd=np.where(ok, Dtu[d][si, np.clip(tt-H0,0,NH-1)], 0.0)
    pp=np.where(ok, Ptu[d][si, np.clip(tt,0,24)], 0.0)
    return dd, pp
def errreg(ev, drug, lags=(1,2,3)):
    m=fc[(fc.event==ev)&(fc.hour>=1)]
    si=m.stay_id.map(sidx_tu).values; t=np.round(m.hour.values).astype(int)
    e=(m.observed-m.predicted_mean).values
    cols=[];names=[]
    for L in lags:
        dd,pp=dosemat(drug,si,t,L); cols+=[pp, dd-pp]; names+=['pi%d'%L,'eps%d'%L]
    X=np.column_stack(cols+[np.ones(len(e))])
    b,res,rk,sv=np.linalg.lstsq(X,e,rcond=None)
    r=e-X@b; s2=r.var(); se=np.sqrt(np.diag(np.linalg.inv(X.T@X))*s2)
    print(ev, drug, 'n',len(e),'err mean %.2f sd %.2f'%(e.mean(),e.std()))
    print('   '+'  '.join('%s %.2f(%.2f)'%(n,bb,ss) for n,bb,ss in zip(names+['c'],b,se)))
for ev in ['glucose_finger','glucose_lab']: errreg(ev,'insulin')
for ev in ['map_arterial','map_cuff']: errreg(ev,'norepinephrine')
```