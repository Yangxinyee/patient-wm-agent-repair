The readout uses the model's predicted mean at each hour, and the model's own draws are fed back. The per-step gain at the first reading after a dose is therefore what matters. Before spending passes, I want an upper bound on what the action token can do at all. Next I will measure, from pass 15 (u=0), how much removing the token shifts finger-glucose predictions after insulin, as a function of the rate.

```python
fu=pd.read_csv('/work/fwd_15/factual_edited.csv')
mu=fo.merge(fu,on=['stay_id','hour','event','observed'],suffixes=('_o','_e'))
mu['dp']=mu.predicted_mean_e-mu.predicted_mean_o; mu['d']=mu.nll_e-mu.nll_o
gu=mu[mu.event=='glucose_finger'].copy(); gu['h']=gu.hour.astype(int)
gu=gu.merge(ins[['stay_id','h','rate']],on=['stay_id','h'],how='left'); gu['rate']=gu.rate.fillna(0)
gu['rb']=pd.cut(gu.rate,[-0.1,0,1.25,2.25,3.25,20])
print(gu.groupby('rb',observed=True).agg(n=('dp','size'),dp=('dp','mean'),dnll=('d','mean')).round(3))
```