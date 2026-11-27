The combination costs +0.00508 nats, just over the 0.005 limit, so it cannot be submitted as is. The two costs roughly add, and head-0's cost grows about linearly with its scale. That makes head-0 the expensive part: it bought -0.37 of extra readout for about 0.0035 nats. Before choosing a smaller head-0 factor, I need to know how precise these 100-stay cost estimates are; this sets the safety margin for unseen stays. I'll compute the paired per-reading standard error for passes 24 and 26, and the cost by event.

```python
for k_ in [24,26]:
    o_=pd.read_csv('/work/fwd_%d/factual_original.csv'%k_); e_=pd.read_csv('/work/fwd_%d/factual_edited.csv'%k_)
    d_=e_.nll.values-o_.nll.values; sid_=o_.stay_id.values
    st_=pd.Series(d_).groupby(sid_).agg(['sum','count'])
    # stay-clustered SE of the per-reading mean
    m_=d_.mean(); se_cl=np.sqrt(((st_['sum']-m_*st_['count'])**2).sum())/st_['count'].sum()
    print(k_, 'mean', round(m_,5), 'iid SE', round(d_.std()/np.sqrt(len(d_)),5), 'stay-clustered SE', round(se_cl,5),
          pd.Series(d_).groupby(o_.event.values).mean().round(4).to_dict())
```