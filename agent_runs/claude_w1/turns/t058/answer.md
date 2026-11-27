The replica is still structurally off. The tool's means differ from my 64-sample average by 2.6, more than twice the 1.2 that 4-sample noise alone would give. My observed-bin NLL is also worse (1.568 against 1.519), and the per-reading NLLs correlate only 0.86.

Before guessing more variants, I'll check where the error sits. I'll group the discrepancy by code and register presence, by whether the previous bundle was a drug bundle, and by hour. I'll also check the norms of the "last status" rows and include a couple of activation alternatives. If the error is concentrated, for example in first readings or after drug bundles, that points to the missing piece.

```python
err=obs-pm64
fa['err']=err; fa['aerr']=np.abs(err)
fa['prev_drug']=(np.round(hr_[fa.prow.values]%1,3)>0)
fa['dtm']=np.round(dt/60)
print(fa.groupby(['event','has']).aerr.mean().round(2).to_dict())
print(fa.groupby('prev_drug').aerr.mean().round(2).to_dict(), fa.groupby('dtm').aerr.agg(['mean','size']).round(2).head(8).to_dict())
print(fa.groupby(pd.cut(fa.hour,[-2,0,3,8,16,25])).aerr.mean().round(2).to_dict())
print('LS row norms',np.linalg.norm(LS,axis=1).round(3),'RB0 norm',np.linalg.norm(RB_[0]).round(3),'VB0',np.linalg.norm(VB[0]).round(3))
print('signed err by event',fa.groupby('event').err.mean().round(2).to_dict())
```