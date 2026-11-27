`T` was a false lead: `.T` in the name lists is just the tensor transpose attribute, and the `T` dict is unrelated results. So `step_b` takes its weights from the dict it is passed.

The real global leak is in `head_value`, which refers to `Po` and `P_` by name. One of those is `P_`, the head-kill variant, and it may be used whenever no dict is passed. `ctx_fn` likewise defaults to `Po`.

To pin it down I'll run the state function directly for the original, `PS` and `P_` and compare. Different states for `PS` and `P_` mean the core is fine, and the identical NLLs come from `head_value` using `P_` for some tensors. Then I'll rebind `P_` to the original and re-check that the original scores 1.40449.

```python
o=tf_states.__code__; print('tf_states args', o.co_varnames[:o.co_argcount], 'n locals', len(o.co_varnames))
hv=head_value.__code__; print('head_value varnames', hv.co_varnames[:30])
S0=tf_states(PO); SS=tf_states(PS); S_=tf_states(P_)
print(type(S0).__name__, len(S0) if isinstance(S0,(tuple,list)) else '')
for i in range(min(3,len(S0))):
    print(i, tuple(S0[i].shape), 'PS-PO %.4g'%float((SS[i]-S0[i]).abs().max()), 'P_-PO %.4g'%float((S_[i]-S0[i]).abs().max()), 'PS-P_ %.4g'%float((SS[i]-S_[i]).abs().max()))
```