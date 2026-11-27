The slot tokens are almost a fixed direction: one direction holds 184.8 of 192 units of variance, and the rest spans about 20–40 dimensions. So a quiet subspace exists for slot tokens. Using it would still need the u-token's direction in the norm1 output space, which I cannot observe, and it would also amplify the "drug present" offset. So I'll deprioritise that idea.

First a cheaper diagnostic on pass 22. If head-0 ×4 mainly adds a systematic bias to predictions (a constant offset in the state), its cost could in principle be offset. If it adds bundle-specific noise, it cannot. I'll compare mean shift against mean absolute shift of predictions per event.

```python
e22 = pd.read_csv('/work/fwd_22/factual_edited.csv')
dd_ = fo_[['event','observed','predicted_mean','nll']].copy(); dd_['pe']=e22.predicted_mean.values; dd_['dp']=dd_.pe-dd_.predicted_mean
dd_['dnll']=e22.nll.values-dd_.nll
g_ = dd_.groupby('event').agg(mean_dp=('dp','mean'), mad_dp=('dp',lambda s: s.abs().mean()), sd_dp=('dp','std'), dnll=('dnll','mean'),
     res_o=('observed', 'count'))
g_['corr_dp_resid'] = dd_.groupby('event').apply(lambda d: np.corrcoef(d.dp, d.observed-d.predicted_mean)[0,1])
print(g_.round(3).to_string())
```