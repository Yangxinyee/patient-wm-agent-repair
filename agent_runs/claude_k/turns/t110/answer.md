The empty lists are my own bug, not missing data. In run 93 the loop variable `for ev, g in M.groupby(...)` overwrote the snapshot dict `g` that I was querying. I'll redo the listing against `globals()` directly to find the DataFrames and large arrays I can reuse.

```python
G_ = globals()
dfs = [(k, G_[k].shape) for k in vars_ if isinstance(G_.get(k), pd.DataFrame)]
arrs = [(k, G_[k].shape) for k in vars_ if isinstance(G_.get(k), np.ndarray) and G_[k].size > 50000]
print('DataFrames:', dfs)
print('big arrays:', arrs)
```