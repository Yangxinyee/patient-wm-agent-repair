Still about 0.8 relative error for every wiring, so something more basic is off. It could be my context vector, the encoders, or what the stored z actually is.

Three diagnostics separate these:
1. Per-dimension correlation between the stored z and my z_post output. Good correlation with wrong scale would point to a normalisation or scaling issue.
2. The error of the prior mean computed from my context against the stored z. If the posterior usually sits near the prior, this tests the context alone.
3. Cross-validated linear R² of the stored z on my [ctx, e, u] inputs.
- High R² means my inputs carry the right information and the MLP usage is wrong.
- Low R² means the inputs themselves are wrong.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
inp=np.concatenate([CI,EI,UI],1)
o_=M2(inp,'z_post','gelu'); zp=o_[:,:32]
cors=[np.corrcoef(zp[:,i],ZT[:,i])[0,1] for i in range(32)]
print('per-dim corr z_post vs stored: median',round(np.median(cors),3),'min',round(np.min(cors),3))
print('scale stored',round(ZT.std(0).mean(),3),'mine',round(zp.std(0).mean(),3), 'mean abs stored',round(np.abs(ZT).mean(),3))
pr_=M2(CI,'z_prior','gelu')[:,:32]
print('prior(ctx w/o gap) err',round(np.abs(pr_-ZT).mean()/np.abs(ZT).mean(),3))
pr2=M2(CI+M2(GI[:,None],'gap','gelu'),'z_prior','gelu')[:,:32]
print('prior(ctx w gap) err',round(np.abs(pr2-ZT).mean()/np.abs(ZT).mean(),3))
for nm,F in [('ctx',CI),('e',EI),('u',UI),('all',inp)]:
    p_=cross_val_predict(Ridge(alpha=10.0),F,ZT,cv=5)
    print(nm,'linear R2',round(1-((ZT-p_)**2).sum()/((ZT-ZT.mean(0))**2).sum(),3))
```