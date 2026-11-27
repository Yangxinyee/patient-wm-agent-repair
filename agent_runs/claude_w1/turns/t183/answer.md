The kernel functions were defined via exec, so their source can't be read back. I'll learn their contracts by inspecting code-object names and calling them on one stay, printing only types and shapes. If gen_samples returns (code, stay, hour, prev, cur, dose lags, with_code) tuples, as the notebook says, I can rebuild the factual and perturbed design matrices directly.

```python
print(type(G3).__name__, len(G3), list(G3.keys())[:2] if isinstance(G3,dict) else '')
print('setA0 in G3:', setA[0] in G3 if isinstance(G3,dict) else '?')
for fn in ['mvdr','gen_samples','run_rows_wc','head_mean','run_stay']:
    co = globals()[fn].__code__
    print(fn, 'vars', co.co_varnames[:25], '| names', co.co_names[:25])
s0 = list(G3.keys())[0]
out0 = gen_samples(G3[s0], pert=False)
print(type(out0).__name__, len(out0))
if len(out0): 
    t0 = out0[0]; print(type(t0).__name__, len(t0) if hasattr(t0,'__len__') else '', [ (type(x).__name__, getattr(x,'shape',None)) for x in t0][:10])
```