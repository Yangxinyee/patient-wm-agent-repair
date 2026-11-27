`tf_eval` also gives identical results for `PS` and `P_`, so the leak is shared. The likely culprits are the layer helpers `step_b` calls: `LN`, `MH`, `FF` and `AGE`, each taking a dict. One of them may index a global dict instead of its argument.

I'll inspect their referenced names, then test directly. If one of them names a global dict, I'll redefine it (in a file) to use its argument and re-check that `PS` ≠ `P_` and that the original scores the same as before.

```python
for f in [LN, MH, FF, AGE, mlpP, LNp, mha, ffn, age_emb_P]:
    c=f.__code__
    print(f.__name__, c.co_varnames[:c.co_argcount], '| names', c.co_names[:25], '| defaults', f.__defaults__ and [type(d).__name__ for d in f.__defaults__])
```