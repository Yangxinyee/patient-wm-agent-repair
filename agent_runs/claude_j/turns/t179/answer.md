Several helpers already do pieces of the recursion: `one_step_err` takes the tool's previous state from `Xt` and applies my `step`, and `run_variant3` runs a stay from `st0`. `bvd` looks like the bundle-vector builder or cache.

I will check what `bvd` and `st0` are, and what `step` returns when called on the first bundle with the tool's z. That gives me the pieces for a clean full recursion with my own z posterior. Output is shapes only.

```python
print(type(bvd).__name__, type(st0).__name__ if 'st0' in globals() else None)
if isinstance(st0,(tuple,list)): print([tuple(t.shape) if hasattr(t,'shape') else type(t).__name__ for t in st0])
ev=BUN[s0][0][1]
r=bvd(ev) if callable(bvd) else bvd[(s0,0)]
print(type(r).__name__, [tuple(t.shape) if hasattr(t,'shape') else type(t).__name__ for t in (r if isinstance(r,(tuple,list)) else [r])])
```