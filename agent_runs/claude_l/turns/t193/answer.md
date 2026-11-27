The training run worked: 4,000 stays, 189k readings and 57k drug bundles (36.5k with insulin, 32.4k with norepinephrine). Mean NLL is 1.608, higher than the tuning batch, which is just a different mix of stays.

This run asks two questions.
1. **Expected dose from context.** How well does a ridge regression on the 192-dim context predict the recorded rate when a record exists (cross-validated by stay)?
2. **Error regression.** Where is the model's error? I regress glucose errors (finger, lab) on insulin residual lags 1–12, and MAP errors (arterial, cuff) on norepinephrine residual lags 1–12. As a check, I then add the expected-dose lags as extra regressors.

What I expect, from my earlier GBM-based analysis:
- Lag-1 and lag-2 insulin coefficients negative and large, roughly −1.4 and −1.1, meaning the model under-reacts.
- Possibly positive coefficients at later lags.
- Expected-dose coefficients near zero, meaning the model has absorbed the policy.

If the expected-dose lags come out large, the model has not absorbed the policy the way my context-based expected dose describes it. The correction would then need a different reference dose.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
meta=DA_meta; ctxA=DA_ctx.astype(np.float64); groups=meta[:,0].astype(int)
PIH={}
for drug,col,hcol in [('ins',2,4),('ne',3,5)]:
    m=meta[:,hcol]==1; Xd=ctxA[m]; yd=meta[m,col]; gd=groups[m]; pred=np.zeros(m.sum())
    for trn,tst in GroupKFold(5).split(Xd,yd,gd):
        pred[tst]=Ridge(alpha=1.0).fit(Xd[trn],yd[trn]).predict(Xd[tst])
    full=Ridge(alpha=1.0).fit(Xd,yd); PIH[drug]=full
    print(drug,'CV R2 %.3f resid sd %.3f y sd %.3f'%(1-((yd-pred)**2).mean()/yd.var(),np.std(yd-pred),yd.std()))
    PIH[drug+'_res']=(m,yd-pred,pred)
# hourly residual / pihat arrays
NS=len(sel); HR=np.floor(meta[:,1]+1e-6).astype(int)
res_h={d:np.zeros((NS,26)) for d in ['ins','ne']}; pih_h={d:np.zeros((NS,26)) for d in ['ins','ne']}
for d in ['ins','ne']:
    m,rr,pp=PIH[d+'_res']; ii=groups[m]; hh=HR[m]
    res_h[d][ii,hh]=rr; pih_h[d][ii,hh]=pp
ri=RA[:,0].astype(int); rt=RA[:,1]; rc=RA[:,2].astype(int)
val=np.array([RA[j,4]*MED[c][1]+MED[c][0] for j,c in enumerate(rc)]); err=val-RA[:,5]
def lagreg(code,drug,L=12,with_pi=False):
    m=(rc==code)&(np.abs(rt-np.round(rt))<1e-6)&(rt>=1)
    i_=ri[m]; t_=np.round(rt[m]).astype(int); y=err[m]
    cols=[np.where(t_-k>=0,res_h[drug][i_,np.maximum(t_-k,0)],0) for k in range(1,L+1)]
    if with_pi: cols+=[np.where(t_-k>=0,pih_h[drug][i_,np.maximum(t_-k,0)],0) for k in range(1,4)]
    Xr=np.column_stack([np.ones(len(y))]+cols); b,*_=np.linalg.lstsq(Xr,y,rcond=None)
    e_=y-Xr@b; se=np.sqrt(np.diag(np.linalg.inv(Xr.T@Xr))*e_.var())
    return b[1:],se[1:],m.sum(),y.std()
for code,drug in [(6,'ins'),(10,'ins'),(7,'ne'),(5,'ne')]:
    b,se,n,sd=lagreg(code,drug); print(code,drug,'n',n,'err sd %.2f'%sd); print('  coef',np.round(b,2)); print('  se  ',np.round(se,2))
    b2,se2,_,_=lagreg(code,drug,L=3,with_pi=True); print('  with pihat lags1-3: res',np.round(b2[:3],2),' pih',np.round(b2[3:],2),'se',np.round(se2[3:],2))
```