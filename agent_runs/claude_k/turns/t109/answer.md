The cached arrays chain exactly: with_code → W1 → GELU → W2 plus the transition bias reproduces my stored logits to the last digit. Any value-head weight edit can therefore be costed exactly on these 5,521 readings, relative to my recon.

Xi is a stay-hour feature table without stay ids, so it can't be joined to the readings directly. To build probe targets I need, for each reading in the 99 state stays, the insulin and norepinephrine rates at lags 1–3 and the policy-expected rates. I'll look for the raw-record DataFrames I loaded earlier rather than re-reading train.csv, by listing every DataFrame and large array with its shape. If a full train-records frame exists, I'll reuse it.

```python
dfs = [(k, g[k].shape) for k in vars_ if isinstance(g.get(k), pd.DataFrame)]
arrs = [(k, g[k].shape) for k in vars_ if isinstance(g.get(k), np.ndarray) and g[k].size > 50000]
print('DataFrames:', dfs)
print('big arrays:', arrs)
```