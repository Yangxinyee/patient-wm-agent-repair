The leak is now visible. `head_value` refers to the globals `Po` and `P_` directly. `ctx_fn` defaults to `Po`, and `step_b` reads a global dict `T` (12 entries), possibly cached transposed weights.

So offline results depend on which dicts were in those globals, not only on the one I pass. That explains why `PS` and `P_` scored the same: whatever `P_` holds was being used.

I'll inspect `T`'s keys and check whether its tensors match the original or the scaffold. If `T` or `P_` hold scaffold values, I'll rebind `P_` and `T` so the functions follow the passed dict, then re-measure the original and the scaffold.

```python
print('T keys:', list(T.keys()))
for k,v in T.items():
    print(' ',k, type(v).__name__, getattr(v,'shape',None) if not isinstance(v,(list,tuple,dict)) else len(v))
    break
c=head_value.__code__
print('head_value consts:', [x for x in c.co_consts if isinstance(x,(str,int,float))][:40])
print('ctx_fn consts:', [x for x in ctx_fn.__code__.co_consts if isinstance(x,(str,int,float))][:30])
print('step_b consts:', [x for x in step_b.__code__.co_consts if isinstance(x,(str,int,float))][:40])
```