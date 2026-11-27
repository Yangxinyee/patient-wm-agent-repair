The decision is now clear: no head-only edit can work. The lag-1 decoder with unit response has a factual noise sd of ~2 u/h, against an innovation sd of 0.52. Even on the perturbed records its response explains only 45% of the variance. The dose signal must be amplified at its source.

The cleanest source is the prelude attention output: the slots attend ~100% to the u token, so that output is a nearly pure function of the dose. Before designing that edit I need to know two things in the replica. Are the weights read from a modifiable cache, so I can try edits locally without `save_edit`? And how are a bundle's action atoms represented, so I can compute the attention value v(d) on a grid of insulin doses? I'll print the prelude-related keys of the weight dict `W` and the structure of one insulin bundle from `bundles_of_stay`.

```python
print('W type', type(W).__name__, 'n', len(W) if hasattr(W, '__len__') else '-')
pk = [k for k in (W.keys() if isinstance(W, dict) else []) if 'prelude' in k]
print('prelude keys:', [(k, tuple(np.shape(W[k]))) for k in pk][:14])
Bs = bundles_of_stay(grp_tr[st[0]], RU)
print('B type', type(Bs).__name__, 'len', len(Bs))
b0 = Bs[0] if isinstance(Bs, (list, tuple)) else None
ib = [i for i, b_ in enumerate(Bs) if isinstance(b_, dict) and any('insulin' in str(v) for v in b_.values())][:1] if isinstance(Bs, list) else []
bb_ = Bs[ib[0]] if ib else b0
if isinstance(bb_, dict):
    print('bundle keys:', {k: (type(v).__name__, np.shape(v) if hasattr(v, 'shape') or isinstance(v, (list, tuple)) else v) for k, v in bb_.items()})
else:
    print('bundle repr:', str(bb_)[:600])
```