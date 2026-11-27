fwd_2 was a factual pass on 100 training stays (`selT`). It gives the baseline value NLL, 1.48105 nats per reading. Per event: glucose_finger 1.449 (MAE 11.0), glucose_lab 1.239 (MAE 9.3), map_arterial 1.381 (MAE 3.9), map_cuff 1.843 (MAE 5.9). This NLL is over the 17 fine bins.

Now the decisive test between (a) and (b). For each glucose reading at hour t+1 that follows an insulin infusion started at t+5min, I'll take the dose deviation, dev = rate − E[rate | decision reading]. The expectation is estimated from all 30k training stays in 5 mg/dL reading bins. I'll regress the model's residual (observed − predicted mean) on dev_t and dev_{t−1}.

- **(a) underfitting:** the slope on dev_t is about −1.6.
- **(b) local fit with flat extrapolation:** the slope is about 0.

I'll run the analogous check for norepinephrine on the next MAP. There the deviation is the rate's log-change against the policy's expected step given the MAP reading and previous rate.

```python
tr = train
g = tr[tr.event.isin(['glucose_finger','glucose_lab'])].copy(); g['h']=g.hour.round().astype(int)
gfv = g[g.event=='glucose_finger'].drop_duplicates(['stay_id','h']).set_index(['stay_id','h']).value
glv = g[g.event=='glucose_lab'].drop_duplicates(['stay_id','h']).set_index(['stay_id','h']).value
dec = gfv.combine_first(glv).rename('R')
ii = tr[tr.event=='insulin'].copy(); ii['h']=np.floor(ii.hour+1e-6).astype(int)
ii = ii.drop_duplicates(['stay_id','h']).set_index(['stay_id','h']).rate.rename('D')
J = pd.concat([dec, ii], axis=1).dropna(subset=['R'])
J['D'] = J['D'].fillna(0.0)
J['rb'] = (J.R//5)*5
ED = J.groupby('rb').D.mean(); J['ED'] = J.rb.map(ED); J['dev'] = J.D - J.ED
print("E[D|R] by band:", {int(k):round(v,2) for k,v in ED.items() if 140<=k<=320 and k%10==0})
print("dev sd (treated):", J[J.D>0].dev.std())
# factual residuals fwd_2 (train stays)
F2 = fa2[fa2.event.isin(['glucose_finger','glucose_lab'])].copy()
F2['h']=F2.hour.round().astype(int); F2['res']=F2.observed-F2.predicted_mean
Jd = J['dev']
F2['dev1'] = [Jd.get((s,h-1), np.nan) for s,h in zip(F2.stay_id,F2.h)]
F2['dev2'] = [Jd.get((s,h-2), np.nan) for s,h in zip(F2.stay_id,F2.h)]
Dd = J['D']
F2['D1'] = [Dd.get((s,h-1), np.nan) for s,h in zip(F2.stay_id,F2.h)]
q = F2.dropna(subset=['dev1','dev2']); q = q[q.D1>0]
import statsmodels.api as smx if False else None
X_ = np.c_[np.ones(len(q)), q.dev1, q.dev2]; bb,_,_,_ = np.linalg.lstsq(X_, q.res.values, rcond=None)
rr = q.res.values - X_@bb; se = np.sqrt(np.diag(np.linalg.inv(X_.T@X_))*rr.var())
print("insulin: n",len(q),"coef(const,dev_t,dev_t-1)",np.round(bb,2),"se",np.round(se,2), "mean res", q.res.mean().round(2))
```