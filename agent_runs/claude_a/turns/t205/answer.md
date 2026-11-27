The dose-edit section settles the evaluation semantics:
- From hour 12 on, every insulin atom has its rate and dose doubled, which changes its z-scores and bins.
- Only glucose and MAP values are drawn by the model; the bundle structure stays as recorded.
- The readout is the predicted mean of the value head (Σ p·centre) at glucose_finger readings in hours 13–24, edited minus factual, averaged over 20 common-random-number paths.

The bytecode of `head_value` shows it calls the global helper `mlp2` for `gap`, `z_prior` and `attr_heads.value_bin`. So I can capture `with_code` (the value-MLP input) by temporarily wrapping `mlp2`, without re-implementing the head. It does not add `last_status` or `bins.rate` for measurement queries, so those must be zero or irrelevant there.

This run:
- prints `mlp2`'s signature and what `tf_eval` calls;
- wraps `mlp2` to record value-head inputs during `tf_eval(PO, dsE)`;
- prints the captured shapes and a sample row of `dsE['rd']`, to find the code column.

If the capture works, the next step is to measure whether `with_code` separates glucose_finger from other codes with a wide linear margin. A wide margin would mean a free code gate.

```python
print('mlp2', inspect.signature(mlp2), '| tf_eval names', tf_eval.__code__.co_names[:40])
_orig_mlp2=mlp2; CAP={}
def _cap(*a, **kw):
    pre=[x for x in a if isinstance(x,str)]
    xs=[x for x in a if torch.is_tensor(x) and x.dim()>=2]
    if pre and xs: CAP.setdefault(pre[0],[]).append(tuple(xs[0].shape))
    return _orig_mlp2(*a, **kw)
globals()['mlp2']=_cap
try:
    r_=tf_eval(PO, dsE, False)
finally:
    globals()['mlp2']=_orig_mlp2
print({k:(len(v), v[:2]) for k,v in CAP.items()})
print('tf_eval out', [tuple(x.shape) if hasattr(x,'shape') else type(x) for x in (r_ if isinstance(r_,tuple) else [r_])])
print('rd row0', np.round(np.asarray(dsE['rd'][0],dtype=float),3), 'nll mean', float(np.mean(r_[0])) if isinstance(r_,tuple) else '')
```