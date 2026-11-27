Run 252 failed inside my own helper chain (`u_rec` → `setenc_my` → `mlpf`) with "list object is not callable". So a global name that `mlpf` uses as a function has been overwritten by a list in recent runs, most likely by `g`, which I reused in run 237. Since the source code isn't retrievable, I'll read the global names referenced by these helpers from their code objects and show each one's current type. If one of them is now a list (for example `g`), I'll restore it to the right function (probably the GELU helper) and re-run the attention measurement.

```python
for f in ['mlpf','setenc_my','u_rec','datom','LNf']:
    co=globals()[f].__code__
    print(f, 'globals used:', [(n, type(globals().get(n)).__name__) for n in co.co_names if n in globals()])
```