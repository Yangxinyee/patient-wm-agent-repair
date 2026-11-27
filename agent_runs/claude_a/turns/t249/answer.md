`ds_edit` failed because lib3.py resets its global `G` to None when it is executed. My earlier `G=globals()` was overwritten when I re-ran the library files. Setting `G` again after loading fixes it. I'll repeat the same decoder fit unchanged.

```python
G=globals()
ds2x=ds_edit(dsE,'insulin',2.0); dsHf=ds_edit(dsE,'insulin',0.5)
keep=[j for j in range(192) if j not in (jU,jR)]
Us=[dsE['u'].numpy(), ds2x['u'].numpy(), dsHf['u'].numpy()]; fs=[1.,2.,.5]
Xd=np.concatenate([np.concatenate([Uk[drug][:,keep],np.ones((drug.sum(),1))],1) for Uk in Us])
yd=np.concatenate([rsI[drug]*f for f in fs])
bd_=ridge_fit(Xd,yd,lam=1e-5)
for Uk,f in zip(Us,fs):
    p_=np.concatenate([Uk[drug][:,keep],np.ones((drug.sum(),1))],1)@bd_; t_=rsI[drug]*f; res=p_-t_; on=t_>0
    sl=np.polyfit(t_[on],p_[on],1)
    print('x%.1f R2 %.4f resid %.3f | starts: resid %.3f slope %.3f max|res| %.2f | NEonly p99 %.3f'%(f,1-res.var()/t_.var(),res.std(),res[on].std(),sl[0],np.abs(res[on]).max(),np.percentile(np.abs(p_[(ne&~ins)[drug]]),99)))
```