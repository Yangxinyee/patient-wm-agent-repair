Run 198 failed on a syntax error in a pointless first line I had left in. I'll remove it and run the same residual regressions unchanged.

```python
def ols(y,X,names):
    X_=np.column_stack(X); c_,_,_,_=np.linalg.lstsq(X_,y,rcond=None); r_=y-X_@c_
    s2=r_.var(); cov=s2*np.linalg.inv(X_.T@X_); print(' '.join('%s %.3f(%.3f)'%(n,c,np.sqrt(cv)) for n,c,cv in zip(names,c_,np.diag(cov))),'| resid sd %.3f'%r_.std()); return c_
tg=TG.copy(); tm=TM.copy()
print('GF rows',len(tg), tg[['r1','r2','r3','m1','m2','m3']].describe().loc[['mean','std']].round(2).to_dict())
one_g=np.ones(len(tg))
ols(tg.res.values,[one_g,tg.r1-tg.m1,tg.r2-tg.m2,tg.r3-tg.m3],['c','d1','d2','d3'])
ols(tg.res.values,[one_g,tg.r1,tg.r2,tg.r3,tg.m1,tg.m2],['c','r1','r2','r3','m1','m2'])
one_m=np.ones(len(tm)); on_=((tm.r1>0)|(tm.r2>0)).astype(float).values
print('MAP rows',len(tm),'frac on',on_.mean().round(3))
ols(tm.res.values,[one_m,tm.r1,tm.r2,tm.r3],['c','r1','r2','r3'])
ols(tm.res.values,[one_m,on_,tm.r1,tm.r2],['c','on','r1','r2'])
ols(tm.res.values-3.45*tm.r1.values-0.72*tm.r2.values,[one_m,on_],['c','on'])
```