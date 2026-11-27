Norepinephrine is always on when MAP is at or below 64 and increasingly likely to be off above that. The rate has a wide spread at every MAP (SD about 0.3), and patients at MAP 90+ who are on it still average 0.53. That looks more like titration from the previous rate (wean when MAP is high, escalate when low) than a sliding scale. The crosstab failed only because `D.ne` clashes with the pandas method.

Next I'll redo the transition table with bracket indexing. I'll also model the rate change: regress (rate_t - rate_{t-1}) on the latest MAP among patients already on norepinephrine, and inspect the residual scatter. That shows whether there's an exogenous noise component in the titration, as there was for insulin, to identify norepinephrine's effect on MAP. Changes that are a clean function of MAP plus independent scatter would confirm titration with usable noise.

```python
print(pd.crosstab(pd.cut(D['ne_prev'],[-.01,0,.1,.2,.4,.6,.8,.99,1.0]), pd.cut(D['ne'],[-.01,0,.1,.2,.4,.6,.8,.99,1.0]), normalize='index').round(2))
on = D[(D['ne_prev']>0)&(D['ne']>0)&(D['ne_prev']<1)&(D['ne']<1)].copy()
on['dr'] = on['ne'] - on['ne_prev']
on['ratio'] = on['ne']/on['ne_prev']
print(on.groupby('mbin').agg(n=('dr','size'), dr=('dr','mean'), dr_sd=('dr','std'), lr=('ratio', lambda x: np.log(x).mean()), lr_sd=('ratio', lambda x: np.log(x).std())).round(3))
```