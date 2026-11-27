Run 222 gives the layout. A bundle is a pair (minute, list of atoms), with atoms written as (event, start/end, record row). `vecs` returns the measurement and action encodings e and u, 192 numbers each.

Now the key experiment, which is to amplify the model's own dose representation. At every insulin bundle I will replace u by u + s·(u(2d) − u(d)), using the encoder itself, for s = 0, 2 and 5. Then I will rerun the factual and insulin x2 counterfactual replica on the 100 stays. Before that, I will check how `vecs`, `bvd` and `run_cf` look up the encoder, so the patch actually takes effect.

What would confirm the idea: the counterfactual glucose shift grows roughly in proportion to (1+s), while factual predictions drop in proportion to dose, so the level bias can be compensated from the last glucose reading. If the shift stays near −0.9, the information is lost downstream of u, and only a head-level partial fix remains.

```python
for n in ['vecs','bvd','run_cf','X_of','mod_events']:
    c=globals()[n].__code__; print(n, inspect.signature(globals()[n]), '| G:',[x for x in c.co_names if x in globals()][:14])
print(type(CACHE).__name__, len(CACHE) if hasattr(CACHE,'__len__') else '')
```