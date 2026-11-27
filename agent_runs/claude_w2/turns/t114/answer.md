`zerr_custom` returns a per-bundle error array, which flooded the output; I'll only print summaries from now on. Next I average that array by bundle kind and set it beside my own per-bundle errors. I also break my glucose-bundle errors down by glucose value. If `zerr_custom` really gives about 0.013 on glucose bundles where I get 0.108, I will inspect its bytecode for the differing step. If the gap is concentrated at particular glucose values, the cause is likely bin handling at the edges, since readings and edges are both integers.

```python
r93=np.asarray(r93)
print(len(r93), {k_:round(float(r93[kt==k_].mean()),4) for k_ in np.unique(kt)})
g_=bk[bk.kt=='gf']
print(g_.err.describe().round(3).to_dict())
print(g_.groupby(pd.cut(g_.gfv,[0,100,150,200,250,300,600]),observed=False).err.agg(['mean','count']).round(3).to_string())
ed_gf=np.array(NUMD[('glucose_finger','value')]['fine_bin_edges'])
g_=g_.assign(onedge=np.isin(g_.gfv,ed_gf))
print(g_.groupby('onedge').err.agg(['mean','count']).round(3).to_string())
```