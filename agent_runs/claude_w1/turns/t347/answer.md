With both probes in, the per-reading decoder values on the tool can be computed. For each finger and arterial reading, the tool's decoder value is the P1 shift divided by the P2 shift. I'll summarise it by:
- its mean (offset) and spread, compared with the replica's spread of 0.70;
- how the spread splits between and within stays;
- its pattern over stay-hours, including the suspicious hours 0–3;
- whether it correlates with the original model's residual, which would mean it carries real signal.

A large mean with a modest spread means the offset can be cancelled in the bias and the edit can work. A spread far above 0.7, or varying by stay, means the decoder is mostly noise on the tool for that drug.

```python
f1o=pd.read_csv('/work/fwd_9/factual_original.csv'); f1e=pd.read_csv('/work/fwd_9/factual_edited.csv')
f2o=pd.read_csv('/work/fwd_10/factual_original.csv'); f2e=pd.read_csv('/work/fwd_10/factual_edited.csv')
D=f1o[['stay_id','hour','event','observed']].copy(); D['mu0']=f1o.predicted_mean
D['a1']=f1e.predicted_mean-f1o.predicted_mean; D['a2']=f2e.predicted_mean-f2o.predicted_mean
print('orig means equal across passes', float((f1o.predicted_mean-f2o.predicted_mean).abs().max()))
for ev_ in ['glucose_finger','map_arterial']:
    x=D[D.event==ev_].copy(); x['d']=x.a1/x.a2
    print(ev_,'a2 pct1/50/99',np.round(np.percentile(x.a2,[1,50,99]),4),'d pct',np.round(np.percentile(x.d,[1,10,25,50,75,90,99]),2),'mean %.2f sd %.2f'%(x.d.mean(),x.d.std()))
    sm=x.groupby('stay_id').d.mean(); print('  between-stay sd %.2f within-stay sd %.2f'%(sm.std(), (x.d-x.stay_id.map(sm)).std()))
    g_=x.groupby(pd.cut(x.hour,[-1,1,3,6,12,18,24]),observed=True).d.agg(['mean','std','size']).round(2)
    print('  by hour', ' | '.join(f'{i}:{r["mean"]},{r["std"]},{int(r["size"])}' for i,r in g_.iterrows()))
    res=x.observed-x.mu0; print('  corr(res,d) %.3f'%np.corrcoef(res,x.d-x.d.mean())[0,1])
    D.loc[x.index,'d']=x.d
```