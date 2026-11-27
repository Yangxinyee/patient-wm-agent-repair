`memvec(j, e_, u_)` chooses what goes into memory from a tuple of option names that my printout truncated. The generator flags drug-only bundles as skippable. I'll print the full option tuple and `memvec`'s constants, to find the variant matching the architecture text ("memory of the last 128 bundle vectors e").

```python
mv=[c_ for c_ in run_variant3.__code__.co_consts if hasattr(c_,'co_code') and c_.co_name=='memvec'][0]
print([x for x in mv.co_consts if not hasattr(x,'co_code')])
print('freevars', run_variant3.__code__.co_cellvars, mv.co_freevars)
```