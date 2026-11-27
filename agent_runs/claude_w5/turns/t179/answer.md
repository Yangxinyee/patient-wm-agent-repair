A bundle is a tuple `(time in hours, list of atoms)`. Drug atoms carry `rate` and `dose`, so the dose edit scales both.

To design the injection I need values inside the core step that `cstep` does not return: h_new (before `c_out` and the coda) and the coda's sensitivity. Since `getsource` fails, I will read `cstep`'s code-object metadata instead: the global names it calls, its local variables and its constants. That should show its exact structure and the weight keys it uses, so I can write a hooked copy and check it against `cstep`.

It will be confirmed if the names reveal its helpers (e.g. a layer function) and keys such as `z_post` and `gap`.

```python
co=cstep.__code__
print('names',co.co_names)
print('vars',co.co_varnames)
print('consts',[c for c in co.co_consts if isinstance(c,(str,int,float))][:60])
rest=sorted(set('.'.join(n.split('.')[:2]) for n in NAMES))
print([r for r in rest if not r.startswith(('attr_heads','embed','encode','core.blocks'))])
```