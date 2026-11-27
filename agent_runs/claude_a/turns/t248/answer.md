Both encoders have norm weights close to 1 in every dimension, so no dimension is especially cheap on the encoder side. I'll use the cheap state dimensions: jU=4 (u-channel and staging), jR=2 (reference), jI=67 (insulin memory).

The u-channel relies on a linear decoder of the start rate from u. It must stay linear when doses are doubled or halved, because the evaluation pushes rates outside the recorded range. So I'll fit it on the original u's together with re-encoded u's at insulin ×2 and ×½, excluding dimensions jU and jR, and check accuracy separately at each scale.

- **Confirms:** R² ≳0.99 and small error at ×2 as well as ×1 and ×½.
- **Refutes:** the encoder saturates at high rates, which would show as a large, rate-dependent error at ×2.

```python
jU,jR,jI=4,2,67
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