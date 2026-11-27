The architecture specifies that bundles without drugs use an exactly zero action vector. That gives a concrete explanation to test for the observation-only mismatch. I’ll inspect the reconstruction helpers and their inputs; replacing an incorrectly encoded empty action with zero should eliminate the mismatch if this is the cause.

```python
import inspect

print("=== RECONSTRUCTION HELPERS ===")
for name in sorted(list(globals())):
    obj = globals()[name]
    if callable(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "?"
        print(name, sig)
        if any(s in name.lower() for s in ("post", "atom", "encode", "bundle", "action", "obs")):
            code = getattr(obj, "__code__", None)
            if code:
                print("  uses:", code.co_names)

print("\n=== RECONSTRUCTION DATA ===")
for name in sorted(list(globals())):
    if any(s in name.lower() for s in ("post", "feat", "bundle", "state", "action", "ref", "obs")):
        obj = globals()[name]
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif hasattr(obj, "shape") and not callable(obj):
            print(name, type(obj).__name__, obj.shape)
        elif isinstance(obj, dict):
            print(name, "dict", list(obj.keys())[:15])
        elif isinstance(obj, list):
            print(name, "list", len(obj))

print("\n=== ENCODER AND POSTERIOR TENSORS ===")
for name in NAMES:
    if name.startswith(("encode.", "encode_action.", "z_post.")):
        print(name, SHAPES[name])
print("Current edits:", edited())
```
