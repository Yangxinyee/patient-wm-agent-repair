The pieces are in place:

- **Teacher-forced training file:** context and u vectors for 57,239 drug bundles (`DA_ctx`, `DA_u`, plus `DA_meta`).
- **Rollout replica:** it computes z from whatever z_post/z_in weights are in the dict `W`, so I can test real weight edits offline.
- **Policy decoders:** linear decodes of the expected dose from the context exist for insulin and NE.

The edit has to fit in z_post (Linear 576→192, GELU, Linear 192→64) and z_in (Linear 32→192).

My design keeps every piece linear, plus one robust gate:
- **Copy of u:** pass the PCA coordinates of u through the hidden layer using the exact identity gelu(x) − gelu(−x) = x.
- **Residual:** r = dose decoded linearly from u, minus the policy decoded linearly from the context.
- **Gate:** the residual only fires when that drug is present. This uses large offsets, gelu(±r + B + M·(s−1)), where s is a linear presence score of u. It only needs s to be roughly 0 or 1, not exact.

Before building, I need to know whether the linear policy decode is good enough. Any policy signal it misses leaks into the residual and biases the factual predictions.

Next I'll check the columns of `DA_meta` and, at insulin and NE bundles, compute:
- the R² of the dose against the context decode;
- the residual sd;
- how well u alone gives back the dose (a linear decode of the dose from u).

**Expectation:** the policy decode R² should be well above 0.5 (the full policy model reached 0.82), and the dose decode from u should be nearly exact (R² > 0.99). A weak policy decode would mean I need nonlinear policy features. A poor dose decode would mean I need per-bin handling.

```python
DM=d_['DA_meta']; DC=d_['DA_ctx']; DU=d_['DA_u']
print('DA_meta[:3]', np.round(DM[:3],3), ' col means', np.round(DM.mean(0),3))
from sklearn.linear_model import Ridge
for j,(nm,pi) in enumerate([('ins',pi_i),('ne',pi_n)]):
    rc=2 if j==0 else 4; hc=3 if j==0 else 5
    m=DM[:,hc]>0.5; d=DM[m,rc]; ph=DC[m]@pi[0].numpy()+pi[1]
    r2c=1-np.var(d-ph)/np.var(d)
    rg=Ridge(1e-3).fit(DU[m],d); r2u=rg.score(DU[m],d)
    print(nm,'n',m.sum(),'dose mean %.3f sd %.3f | ctx-decode R2 %.3f resid sd %.3f | u-decode R2 %.5f'%(d.mean(),d.std(),r2c,np.std(d-ph),r2u))
```