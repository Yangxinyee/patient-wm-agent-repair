Insulin is now clearly null. With strong dose-time controls, the clean intensity noise has no effect on finger glucose at any lag from 1 to 12 hours. Every coefficient is within ±0.12 mg/dL per unit, and none is significant. A shock also does not carry into later doses (persistence ≈ 0 beyond lag 0), so no feedback needs correcting.

The norepinephrine part crashed only because `runA` was never defined, since the earlier run died before reaching that line.

This run:
- Defines `runA` and reruns the norepinephrine impulse responses and their persistence in later rates.
- Converts both drugs' impulse responses into implied x2 and x1/2 readouts over hours 13–24, with lower and upper SE bounds.
- For norepinephrine, first deconvolves the MAP responses by the persistence to get per-hour direct effects.

Implied shifts near 0 (within about ±0.5) for both drugs would confirm that the zero-effect edit matches the truth.

```python
csA=np.cumsum(np.nan_to_num(Ar),1); ccA=np.cumsum(~np.isnan(Ar),1); runA=np.where(ccA>0,csA/np.maximum(ccA,1),np.nan)
bN1,sN1=impulse(Ar,NA,okN,cellfnN,ctrl_ne,label='NE NA -> art MAP')
pN,_=impulse(Ne,NA,okN,cellfnN,ctrl_ne,K=11,label='NE NA -> later NE rate (persistence)')
def implied(beta, se, drug, Y, factor=1.0, label=''):
    # x2: adds drug_s for s>=12 ; half: subtracts drug_s/2. beta[k] = direct effect of unit change at lag k (k=1..12)
    cs=[]
    for t in range(13,25):
        ti=t+1; D=np.stack([drug[:,ti-k] if k<=t-12 else np.zeros(n) for k in range(1,13)],1)
        reached=(D>0).any(1)&~np.isnan(Y[:,ti])*0+((D>0).any(1))
        cs.append(D[(D>0).any(1)])
    C=np.concatenate(cs); c=C.mean(0)
    b=beta[1:13]; s=se[1:13]
    est=c@b; se_ind=np.sqrt((c**2*s**2).sum()); se_full=(c*s).sum()
    print('%s x2: %+.3f (se %.3f..%.3f)  half: %+.3f'%(label,est,se_ind,se_full,-0.5*est), ' c=',np.round(c,2))
    return est
implied(bI1,sI1,Ins,Fg,label='INSULIN direct')
# NE deconvolution: beta_k = sum_{l=1..k} delta_l * rho_{k-l}
rho=pN; K=12; delta=np.zeros(K+1)
for k in range(1,K+1):
    delta[k]=bN1[k]-sum(delta[l]*rho[k-l] for l in range(1,k))
print('NE rho:',np.round(rho,3)); print('NE delta (direct per-hour effects):',np.round(delta[1:],3))
implied(delta,sN1,Ne,Ar,label='NE direct (deconvolved)')
implied(bN1,sN1,Ne,Ar,label='NE total-effect naive')
```