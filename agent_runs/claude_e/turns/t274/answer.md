The offline run failed only on a dtype mismatch. `head_run` evidently expects float64 weight tensors, which is what `gated3`/`ins_only` produce, while `load_original` returns float32. I will convert the original weights to float64 and rerun the small probe to learn the output format.

```python
print({k: v.dtype for k,v in gi.items()}, X.dtype)
orig = {k: load_original(n).double() for k,n in KMAP.items()}
t=time.time(); o = head_run(X, R1.iloc[:200], orig); print('t', round(time.time()-t,2))
print(type(o))
if isinstance(o, dict): print({k: getattr(v,'shape',type(v)) for k,v in o.items()})
elif isinstance(o, (tuple,list)): print([getattr(v,'shape',type(v)) for v in o])
else: print(getattr(o,'shape',None))
```