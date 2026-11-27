That printout was wasteful: it dumped the default weight dictionary of my `prelude` replica. It still showed what I needed about `step_full`. It takes E (the prelude output), the time‑decayed h (`htf`) and row indices. It runs the recurrent block (self‑attention, memory cross‑attention, FFN), the input/decay update (`delta_in`, `a_in`, `b_in`), `c_out`, the coda and `out_norm`. `dE` is a perturbation of E.

So I have the pieces of a full CPU chain: prelude replica → `step_full` → my new value‑head replica. With it I could design the injection on CPU and use forward passes only to verify. To call them correctly I need:
- `prelude`'s `ret` options and token layout;
- the drug‑bundle token arrays, row index mapping and htf arrays.

The earlier array listing was cut off at "allA", so next I'll print the rest of the array names (names and shapes only), `prelude`'s non‑weight defaults and its globals. If I find a token array of shape (2319, 7, 192) and a matching row‑index array into the states rows, the chain is callable. If not, I'll fall back to empirical probing.

```python
arr=sorted((k,tuple(v.shape)) for k,v in list(globals().items()) if not isinstance(v,types.ModuleType) and isinstance(getattr(v,'shape',None),tuple) and not k.startswith('_'))
names=[a for a in arr if a[0]>='allA']
print(names)
co=prelude.__code__; print('prelude args',co.co_varnames[:co.co_argcount],'ret default',prelude.__defaults__[-1] if not isinstance(prelude.__defaults__[-1],dict) else 'dict')
print(' locals',co.co_varnames[co.co_argcount:co.co_argcount+25]); print(' globals',co.co_names[:40]); print(' consts',[c for c in co.co_consts if isinstance(c,(str,int,float))][:30])
for nm in ['build_mem','age_emb','mha_m','tok_with','toks','u_bundle','utok','u_for']:
    f=globals().get(nm); co=f.__code__; print('==',nm,co.co_varnames[:co.co_argcount],'| globals',co.co_names[:25],'| consts',[c for c in co.co_consts if isinstance(c,(str,int,float))][:12])
```