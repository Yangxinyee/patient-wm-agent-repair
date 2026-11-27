The factual pass rules out v7 at these gains. Value NLL rose by +0.254 nats per reading, about 50 times the 0.005 budget:

| Readings | NLL change | Driven by |
|---|---|---|
| Finger glucose | +0.17 | insulin |
| Lab glucose | +0.39 | insulin |
| Arterial MAP | +0.37 | NE |
| Cuff MAP | +0.18 | NE |

Lab MAE went from 9.1 to 12.7 mg/dL.

So the injection moves predictions at the actual readings far more than its slowly accumulating rollout effect would need. At gain 1, each insulin bundle gives an immediate shift of about −0.94 mg/dL per unit rate. Yet the x2 rollout effect only grows by about 0.33 mg/dL per hour, so most of each pulse is lost before it can be carried forward.

If cost simply scales with gain squared, meeting the budget would leave only about 12% of the true effects. Before choosing between scaling down and redesigning, I need to know where the cost arises. This run therefore:

- splits the factual prediction shifts and NLL changes by reading type;
- relates them, if the record's timing allows, to how soon each reading follows a drug bundle and to the recent dose.

How to read the result:

- If shifts are concentrated in readings right after drug bundles, timing explains the poor ratio, and a redesign aimed at persistence could help.
- If shifts are spread evenly, the trade-off is inherent and I scale the gains down to fit the budget.

```python
fe=pd.read_csv('/work/fwd_12/factual_edited.csv'); fo=pd.read_csv('/work/fwd_12/factual_original.csv')
assert (fe.stay_id.values==fo.stay_id.values).all() and (fe.event.values==fo.event.values).all()
d=fe[['stay_id','hour','event','observed']].copy(); d['dp']=fe.predicted_mean.values-fo.predicted_mean.values; d['dn']=fe.nll.values-fo.nll.values
d['res0']=fe.observed.values-fo.predicted_mean.values
print('hour dtype',d.hour.dtype,'frac nonint %.2f'%(np.mean(d.hour%1!=0)))
g=d.groupby('event')
print(pd.DataFrame({'dnll':g.dn.mean(),'dp_mean':g.dp.mean(),'dp_rms':g.dp.apply(lambda x: np.sqrt((x**2).mean())),'res0_sd':g.res0.std(),'corr(dp,res0)':g.apply(lambda x: np.corrcoef(x.dp,x.res0)[0,1])}).round(3))
tu=pd.read_csv('/data/tuning.csv',nrows=5)
print('tuning cols',list(tu.columns))
print('dtypes',dict(tu.dtypes.astype(str)))
```