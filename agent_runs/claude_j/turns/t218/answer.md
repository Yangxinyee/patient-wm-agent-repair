For this stay the recomputed predictions match the tool at all hours (about 1.6 mg/dL error) whatever the restart point, and the stored states equal the recomputed ones. So stored states are not the general problem. The large early-hour errors must come from particular stays.

Next I break the replica-vs-tool error at hours ≤ 12 down by stay.
- If a few stays carry it (say, ones with unusual first bundles), I either fix the alignment or exclude them.
- If it is spread evenly, something in how I built the factual chunk differs from this direct check.

```python
mm['e']=(mm.mf-mm.predicted_mean).abs()
ps=mm[mm.hour<=12].groupby('stay_id').e.mean().sort_values()
print('stays',len(ps),'quantiles',np.round(ps.quantile([.1,.5,.75,.9,1]).values,2))
print('n stays err>5:',int((ps>5).sum()),'| worst',[(int(k)%100000,round(v,1)) for k,v in ps.tail(4).items()])
s=int(ps.index[-1]); x=mm[(mm.stay_id==s)].sort_values('hour'); print('worst stay hours/err', list(zip(x.hour.values[:10],np.round(x.e.values[:10],1))))
```