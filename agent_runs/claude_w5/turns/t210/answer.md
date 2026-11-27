I now have the interface signatures. The hooked core cstep_h takes inj, rec and dE hooks, and run_tf drives a teacher-forced pass over a stay's bundle list. I don't remember what these return, so next I probe them on a 3-bundle slice:
- the layout of a bundle tuple and of a run-185 `rows` entry;
- what run_tf(..., keep=True) returns;
- which internals cstep_h records into `rec`.

If rec exposes the prelude internals (n1 tokens, n3, P, E) and h_new, I can build the edit's regressions and its Jacobian directly with these. Otherwise I write a small wrapper.

```python
sid0 = list(bundles_B.keys())[0]
B = bundles_B[sid0]
def desc(x):
    if isinstance(x, (int, float, str, bool, np.integer, np.floating)): return f"{type(x).__name__}:{str(x)[:20]}"
    if hasattr(x, 'shape'): return f"{type(x).__name__}{tuple(x.shape)}"
    if isinstance(x, dict): return f"dict{list(x.keys())[:6]}"
    if hasattr(x, '__len__'): return f"{type(x).__name__}[{len(x)}]"
    return type(x).__name__
print('B[0]', [desc(x) for x in B[0]])
print('B[1]', [desc(x) for x in B[1]])
print('rows[0]', [desc(x) for x in rows[0]])
res = run_tf(B, kmax=3, keep=True)
print('run_tf', desc(res))
if isinstance(res, (list, tuple)):
    for x in res[:6]: print('  ', desc(x))
    if len(res) and isinstance(res[0], (list,tuple)): print('   elem0', [desc(y) for y in res[0]][:8])
elif isinstance(res, dict):
    for k, x in list(res.items())[:10]: print('  ', k, desc(x))
```