There's no input history, so the replica and head functions are black boxes. I'll learn how to drive them from their bytecode metadata (the global names they call and their constants) and from the structure of one cached bundle. The goal is a function that recomputes states with altered doses. If `one_step_err` / `run_variant3` show the call order (`bundle_vecs` → `encode` → `step`), I can chain them myself.

```python
def cinfo(f):
    c_=f.__code__
    return 'names:'+','.join(c_.co_names)[:300]+' | consts:'+','.join(str(x)[:15] for x in c_.co_consts if not hasattr(x,'co_code'))[:200]
for f in ['one_step_err','run_variant3','bvd','bundle_vecs','step']:
    print(f, cinfo(globals()[f]))
b0=BUN['2020000000']; print(type(b0), len(b0))
e0=b0[0] if isinstance(b0,list) else b0
print(type(e0), (list(e0.keys()) if isinstance(e0,dict) else str(e0)[:200]))
```