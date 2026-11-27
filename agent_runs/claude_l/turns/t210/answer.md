I used the wrong columns. `DA_meta` is [stay, time, insulin rate, NE rate, has insulin, has NE], so the R² values just printed are meaningless. I'll redo the same check with the correct columns:
- the policy decode from the context;
- the residual sd;
- the dose decode from u.

I'll also fit a fresh ridge policy decode from the context on these bundles, with a train/test split, to see whether the stored `pi_i`/`pi_n` are the best linear decodes available.

**Expectation:** policy decode R² of roughly 0.6–0.9 and dose-from-u R² above 0.99.

```python
from sklearn.model_selection import train_test_split
for nm,pi,rc,hc in [('ins',pi_i,2,4),('ne',pi_n,3,5)]:
    m=DM[:,hc]>0.5; d=DM[m,rc]; X=DC[m]; ph=X@pi[0].numpy()+pi[1]
    r2c=1-np.var(d-ph)/np.var(d)
    stays=DM[m,0]; tr=(stays%5)!=0
    rg=Ridge(1.0).fit(X[tr],d[tr]); r2f=rg.score(X[~tr],d[~tr]); sdf=np.std(d[~tr]-rg.predict(X[~tr]))
    ru=Ridge(1e-3).fit(DU[m],d); r2u=ru.score(DU[m],d)
    print(nm,'n',m.sum(),'dose mean %.3f sd %.3f | stored ctx-decode R2 %.3f sd %.3f | fresh ridge test R2 %.3f sd %.3f | u-decode R2 %.5f'%(d.mean(),d.std(),r2c,np.std(d-ph),r2f,sdf,r2u))
```