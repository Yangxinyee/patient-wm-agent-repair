The bootstrap standard error of the cost over stays is large: 0.00141 on 100 stays. So N5+H2's +0.0037 could plausibly exceed 0.005 on unseen stays. The cost is concentrated in glucose_lab (+0.011) and map_arterial (+0.0064). I'll compute the same paired statistics for the earlier passes: fwd_24 (N5) and fwd_26 (N5+H2.5). That shows how much of the variance comes from the head0 part. I'd expect N5 alone to have a much smaller SE; if so, the safe choice is N5 with only a small head0 boost.

```python
def stat(f):
    e=pd.read_csv(f); d=e.nll.values-b.nll.values; df=pd.DataFrame({'s':e.stay_id,'ev':e.event,'d':d})
    g=df.groupby('s').d.agg(['sum','count']); S=g['sum'].values; C=g['count'].values
    bs=[S[i].sum()/C[i].sum() for i in (rng.integers(0,len(S),len(S)) for _ in range(3000))]
    return round(d.mean(),5), round(np.std(bs),5), df.groupby('ev').d.mean().round(4).to_dict(), g['sum'].sort_values().values[[0,1,-2,-1]].round(2)
import os
for k in [24,26,15,10]:
    f=f'/work/fwd_{k}/factual_edited.csv'
    if os.path.exists(f): print(k, stat(f))
```