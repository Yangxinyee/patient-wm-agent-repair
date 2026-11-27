The architecture confirms separate observation and action encoders, but the variable listing was truncated before reaching the fresh implementation. I’ll inspect only the recent helpers, without printing large tensors. A difference in pooling or normalization would explain the mismatch; identical implementations would point back to inputs or alignment.

```python
import torch, inspect, dis

print("=== FRESH IMPLEMENTATION INVENTORY ===")
for prefix in ("r100", "r101", "r102", "r103", "r104"):
    for name in sorted(n for n in globals() if n.startswith(prefix)):
        obj = globals()[name]
        if callable(obj):
            code = getattr(obj, "__code__", None)
            print(name, str(inspect.signature(obj)), "names=", code.co_names if code else None)
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "dict", len(obj), "keys=", list(obj)[:6])
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, len(obj))
        elif isinstance(obj, (str, int, float, bool, np.integer, np.floating)):
            print(name, repr(obj)[:100])
        else:
            print(name, type(obj).__name__)

print("\n=== ENCODER HELPER SIGNATURES ===")
for name in ("r86_encode_observations", "r79_mlp", "r79_ln"):
    if name in globals():
        fn = globals()[name]
        print(name, inspect.signature(fn), fn.__code__.co_names)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 106\nArchitecture recovered. Run 105 inventory was truncated by tensor representations; restricting output to fresh helper signatures and array shapes. No edits. Counts: 106 Python runs, 6 forward passes.\n")
```