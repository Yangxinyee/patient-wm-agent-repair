The sample records include optional lab and cuff readings, so a measurement bundle cannot always be reconstructed from just glucose_finger and map_arterial. I’ll verify the recurrent tensor layout and context formula before rebuilding bundles from all their recorded events. A match on posterior means is the next required check.

```python
print("=== CORE TENSOR LAYOUT ===")
for n in NAMES:
    if n.startswith("core."):
        print(n, SHAPES[n])

print("\n=== POSTERIOR / CONTEXT WORKSPACE ===")
for n, v in sorted(list(globals().items())):
    if any(s in n.lower() for s in (
        "ctx", "empty", "measure", "world", "posterior", "post_spec", "best_spec"
    )) and not n.startswith("rm_"):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(n, tuple(v.shape))
        elif isinstance(v, (str, int, float, tuple)):
            print(n, repr(v)[:300])
        elif callable(v) and hasattr(v, "__code__"):
            print(n, inspect.signature(v), v.__code__.co_names,
                  repr(v.__code__.co_consts)[:600])

print("\n=== LARGE DATAFRAMES ===")
for n, v in sorted(list(globals().items())):
    if isinstance(v, pd.DataFrame) and len(v) > 10000:
        print(n, v.shape, list(v.columns)[:12])

print("\n=== HELPER CONSTANTS ===")
for n in ("rm_lin", "rm_ln", "rm_mlp"):
    f = globals()[n]
    print(n, f.__code__.co_names, f.__code__.co_consts)
```
