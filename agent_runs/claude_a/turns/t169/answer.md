`tf_nll` crashed inside `head_value` with a shape mismatch (2 vs 5802). It reads several globals (`prevrow`, `dtv`, `cidx`, `hasl`, `ages`, `zl`, `lvbt`, `eps4`, `obs_t`). One of them was probably overwritten by a later run with something of width 2.

I'll print their shapes. Then I can either rebuild the bad one (most likely `ages`) or confirm the kernel functions are no longer safe to use as they are.

```python
for nm in ['prevrow','dtv','cidx','hasl','ages','zl','lvbt','eps4','obs_t','age_h','codes_','obs_pm','centM','gmT','mmT','base0','trans0','Wln','T','A']:
    v=globals().get(nm,'MISSING')
    if isinstance(v,str): print(nm,v); continue
    sh=getattr(v,'shape',None)
    print(nm, type(v).__name__, sh if sh is not None else (len(v) if hasattr(v,'__len__') else v))
print(head_value.__code__.co_varnames[:head_value.__code__.co_argcount])
print(tf_states.__code__.co_varnames[:tf_states.__code__.co_argcount], tf_states.__defaults__ and [type(d).__name__ for d in tf_states.__defaults__])
```