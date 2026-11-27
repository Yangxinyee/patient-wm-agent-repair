The kernel still has the training records (`tr` and `trs`), the tuning records (`tu`), the weight dict `W`, `sel1` and `selT`, and two 12-lag arrays `Kins` and `Knor`. Those are probably my structural impulse responses. Before building the counterfactual experiment I need three things: the column formats (which frame `prep_rollout` expects), what `Kins`/`Knor` hold, and whether my policy function `pfun` still exists. That last one is needed to compute dose deviations. Next I print those and the function names, which were cut off.

```python
import types
fns=[k for k,v in globals().items() if isinstance(v,types.FunctionType)]
print(fns)
print('Kins',np.round(Kins,3)); print('Knor',np.round(Knor,3)); print('Kv',np.round(Kv,3)); print('K',np.round(np.array(K,dtype=float),3) if all(np.isscalar(x) for x in K) else K)
for nm_ in ['tr','trs','tu']:
    df_=globals()[nm_]; print(nm_, df_.shape, list(df_.columns)); print(df_.head(3).to_string())
print(tr.code.value_counts().head(8) if 'code' in tr else '')
```