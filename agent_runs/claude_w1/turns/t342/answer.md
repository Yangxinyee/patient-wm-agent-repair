Progress, but only half the edit works:

| Reading | NLL original → edited | Mean abs error |
|---|---|---|
| glucose_lab | 1.2605 → 1.2603 | unchanged |
| map_cuff | 1.7463 → 1.7468 | unchanged |
| map_arterial (drug-2 part) | 1.3492 → 1.3655 (+0.016) | 3.96 → 4.08 |
| glucose_finger (drug-1 part) | 1.53 → 9.86 | 10.8 → 83.6 |

So removing the decoder offset fixed the gate leakage. The drug-2 part degrades only mildly, which is about 0.004 overall: still too much, but recoverable by lowering its gain. The drug-1 part is still broken.

The drug-1 decoder has norm about 470, against 82 for drug 2, so it is far more sensitive to any replica-vs-tool mismatch in the head input. An average error of about 80 suggests a systematic deviation that drives its unit out of the linear range, not modest noise.

Before choosing between a heavily norm-limited drug-1 decoder and dropping drug 1, I'll look at the per-reading shifts in fwd_8:
- their sign, spread and dependence on stay-hour, for finger and arterial rows;
- whether the arterial shifts line up with the original model's residuals, which is the evidence that the drug-2 part moves predictions in the right direction.

A one-sided, hour-dependent finger shift would point to slow-channel drift in my replica.

```python
fo=pd.read_csv('/work/fwd_8/factual_original.csv'); fe=pd.read_csv('/work/fwd_8/factual_edited.csv')
d=fo[['stay_id','hour','event','observed']].copy(); d['mu0']=fo.predicted_mean; d['dmu']=fe.predicted_mean-fo.predicted_mean; d['dn']=fe.nll-fo.nll
for ev_ in ['glucose_finger','map_arterial']:
    x=d[d.event==ev_]; res=x.observed-x.mu0
    print(ev_,'n',len(x),'dmu pct1..99',np.round(np.percentile(x.dmu,[1,10,25,50,75,90,99]),2),'mean dn %.4f'%x.dn.mean())
    print('  slope res~dmu %.3f'%((res*x.dmu).sum()/(x.dmu**2).sum()),'corr %.3f'%np.corrcoef(res,x.dmu)[0,1],'frac|dmu|>20 %.3f'%(x.dmu.abs()>20).mean())
    g_=x.groupby(pd.cut(x.hour,[-1,3,6,12,18,24,99]),observed=True).dmu.agg(['mean','std','size']).round(2)
    print('  by hour', ' | '.join(f'{i}:{r["mean"]},{r["std"]},{int(r["size"])}' for i,r in g_.iterrows()))
```