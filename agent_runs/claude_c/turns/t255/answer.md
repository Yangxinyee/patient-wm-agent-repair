Five passes saved the original's per-reading factual file for these 4,751 readings, with columns stay_id, hour, event, observed, predicted_mean and nll. I'll pair the N5+H2 rows with the original rows from fwd_2 and confirm they line up. Then I'll bootstrap the per-reading cost over stays to get its standard error, and split the cost by event. The aim is to judge whether +0.0037 stays under 0.005 on unseen stays.

```python
e=pd.read_csv('/work/fwd_28/factual_edited.csv'); b=pd.read_csv('/work/fwd_2/factual_original.csv')
al=(e[['stay_id','hour','event','observed']].values==b[['stay_id','hour','event','observed']].values).all()
d=e.nll.values-b.nll.values; df=pd.DataFrame({'s':e.stay_id,'ev':e.event,'d':d})
g=df.groupby('s').d.agg(['sum','count']); rng=np.random.default_rng(0)
bs=[]; S=g['sum'].values; C=g['count'].values
for _ in range(4000):
    i=rng.integers(0,len(S),len(S)); bs.append(S[i].sum()/C[i].sum())
print(al, round(d.mean(),5), round(np.std(bs),5), df.groupby('ev').d.mean().round(4).to_dict())
```