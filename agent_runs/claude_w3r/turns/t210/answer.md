Run 197 found the factual tables from fwd_3: 5684 values over 100 train stays.

| Reading | Rows | MAE |
|---|---|---|
| glucose_finger | 1865 | 11.0 mg/dL |
| glucose_lab | 541 | 9.2 mg/dL |
| map_arterial | 2481 | 4.0 mmHg |
| map_cuff | 797 | 5.6 mmHg |

TG holds the glucose_finger rows, with residuals, insulin lags r1–r3 and policy band means m1–m3. TM holds the map_arterial rows, with residuals and norepinephrine lags.

MAP is predicted much more tightly than glucose (sd ≈ 5 mmHg). So channel noise matters more for MAP: about 0.7 mmHg of noise would cost about 0.002 nats per value overall, and gating may be worth it there.

Next I'll pin down the compensation forms by regressing the original model's residuals on:
- glucose: insulin deviations from the policy band mean;
- MAP: norepinephrine rate levels with an intercept, plus a variant with an "on norepinephrine" indicator.

The fitted lag-1 and lag-2 coefficients near the record-based effects (−2.0/−0.95 and 3.45/0.72), less the model's partial capture, will fix the compensation term. A large on-indicator coefficient would mean the norepinephrine intercept should apply only while the drug is running.

```python
import statsmodels.api as sm if False else None
def ols(y,X,names):
    X_=np.column_stack(X); c_,res_,_,_=np.linalg.lstsq(X_,y,rcond=None); r_=y-X_@c_
    s2=r_.var(); cov=s2*np.linalg.inv(X_.T@X_); print(' '.join('%s %.3f(%.3f)'%(n,c,np.sqrt(cv)) for n,c,cv in zip(names,c_,np.diag(cov))),'| resid sd %.3f'%r_.std()); return c_
tg=TG.copy(); tm=TM.copy()
print('GF rows',len(tg),'cols sample',tg[['r1','r2','r3','m1','m2','m3']].describe().loc[['mean','std']].round(2).to_dict())
one_g=np.ones(len(tg))
ols(tg.res.values,[one_g,tg.r1-tg.m1,tg.r2-tg.m2,tg.r3-tg.m3],['c','d1','d2','d3'])
ols(tg.res.values,[one_g,tg.r1,tg.r2,tg.r3,tg.m1,tg.m2],['c','r1','r2','r3','m1','m2'])
one_m=np.ones(len(tm)); on_=((tm.r1>0)|(tm.r2>0)).astype(float).values
print('MAP rows',len(tm),'frac on',on_.mean().round(3))
ols(tm.res.values,[one_m,tm.r1,tm.r2,tm.r3],['c','r1','r2','r3'])
ols(tm.res.values,[one_m,on_,tm.r1,tm.r2],['c','on','r1','r2'])
ols(tm.res.values-3.45*tm.r1.values-0.72*tm.r2.values,[one_m,on_],['c','on'])
```