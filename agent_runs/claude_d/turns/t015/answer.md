Norepinephrine follows a titration protocol.
- **Start:** it starts when arterial MAP is below 65, with a higher starting rate the lower the MAP (0.52 below 50, about 0.14 at 62–64). A few starts happen at 65–80, perhaps driven by a cuff reading or noise.
- **Continuation:** once running it almost always continues, and the rate tracks the previous rate closely (r ≈ rprev in every MAP bin). So the hour-to-hour change carries the decision, unlike insulin, where the rate is reset from glucose every hour.

To find a random component for identification, I will look at the hour-to-hour change (difference and ratio) against current MAP. I will also check what drives starts outside the threshold and stops. If there is a random part in the change given MAP and the previous rate, it is the norepinephrine natural experiment.

```python
co = c[(c.nor_prev>0)&c.nor.notna()&c.ma.notna()].copy()
co['ratio'] = co.nor/co.nor_prev; co['diff']=co.nor-co.nor_prev
tab = co.groupby(pd.cut(co.ma,[0,45,50,55,60,62,64,65,66,68,70,72,75,78,80,85,90,200]),observed=True).agg(n=('nor','size'), ratio_mean=('ratio','mean'), ratio_sd=('ratio','std'), diff_mean=('diff','mean'), diff_sd=('diff','std'), lr_sd=('ratio',lambda x: np.log(x).std()))
print(tab.round(3).to_string())
# examples of ratio distribution at MAP<65 and MAP in 70-75
for lo_,hi_ in [(55,60),(70,75),(80,90)]:
    s = co[(co.ma>lo_)&(co.ma<=hi_)]
    print(lo_,hi_, np.round(np.quantile(s.ratio,[0.01,0.05,.1,.25,.5,.75,.9,.95,.99]),3))
# stops
cs = c[(c.nor_prev>0)&c.ma.notna()].copy(); cs['stop']=c.nor.isna()
print(cs.groupby(pd.cut(cs.nor_prev,[0,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.5,1.01]),observed=True).stop.agg(['mean','size']).round(3).to_string())
```