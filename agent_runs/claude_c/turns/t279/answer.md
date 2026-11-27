On stays 200-299 N5 costs +0.00192 (1.61959 → 1.62151). The three batches now give +0.00156, +0.00201 and +0.00192, so the cost is a steady ~+0.0018-0.002. The batches agree far more closely than the per-stay bootstrap (±0.0011) implied, which suggests that bootstrap overstated the risk. That reopens a small head0 boost as a possibly affordable add-on.

Before deciding, I'll estimate how precisely the head0 x2 increment is known. I'll take the paired per-stay difference between N5+H2 (fwd_28) and N5 (fwd_24) on the same stays and bootstrap it. If the increment is about +0.0021 with SE ≤ 0.0005, a head0 factor of about 1.5 (expected total ≈ +0.003) is safe.

```python
e28=pd.read_csv('/work/fwd_28/factual_edited.csv'); e24=pd.read_csv('/work/fwd_24/factual_edited.csv')
dd=pd.DataFrame({'s':e28.stay_id,'ev':e28.event,'d':e28.nll.values-e24.nll.values})
g=dd.groupby('s').d.agg(['sum','count']); S=g['sum'].values; C=g['count'].values
bs=[S[i].sum()/C[i].sum() for i in (rng.integers(0,len(S),len(S)) for _ in range(3000))]
print(round(dd.d.mean(),5), round(np.std(bs),5), dd.groupby('ev').d.mean().round(4).to_dict(), np.sort(S)[[0,1,-2,-1]].round(2))
```