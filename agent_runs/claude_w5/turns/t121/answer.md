The value head is a two-layer MLP: 192 → 192 hidden units → 17 bins. The hidden layer can respond to glucose and MAP queries differently, so a dose memory can be routed to shift only the right reading type. That is better than a fixed tilt of the logits, which for large shifts would wrongly push mass into far bins.

The first building block is the write path. Doses enter only through the action encoder and then the prelude output E. If E already encodes the dose rate linearly, one row of `core.b_in` can write it into a memory dimension.

Next I check how to drive my local reimplementation: the stored options, the bundle format, and what `test_step` returns.

```python
print(type(best_), (best_ if not hasattr(best_,'shape') else best_.shape))
sid0 = list(bundles_B)[0]; bb0 = bundles_B[sid0]
print(type(bb0), len(bb0), type(bb0[0]), (list(bb0[0].keys()) if isinstance(bb0[0],dict) else bb0[0][:3] if isinstance(bb0[0],(list,tuple)) else ''))
try:
    r = test_step([sid0], best_, kmax=3); print('test_step ->', type(r), (r if not isinstance(r,(list,tuple,dict)) else str(r)[:300]))
except Exception as ex: print('ERR', repr(ex)[:300])
```