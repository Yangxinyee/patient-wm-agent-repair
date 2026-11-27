The architecture confirms that the current edit changes only the reading-value head. The diagnostic output was truncated before reaching the estimates and helper functions, so I’ll retrieve those compactly. This should let us test a stronger candidate without rebuilding already-validated machinery.

```python
print("=== EARLY FINDINGS ===")
print(_notes_recovered[:5200])

print("=== KERNEL FUNCTION NAMES ===")
_fn_names = [
    k for k, v in list(globals().items())
    if inspect.isfunction(v) and not k.startswith("__")
]
print(", ".join(_fn_names))

print("=== RECENT VARIABLE NAMES ===")
print(", ".join(list(globals().keys())[-150:]))

print("=== CHECKPOINTS AND EDITS ===")
for k, v in list(globals().items()):
    if "checkpoint" in k.lower():
        if isinstance(v, dict):
            print(k, [(n, tuple(t.shape)) for n, t in v.items()
                      if isinstance(t, (np.ndarray, torch.Tensor))][:6])
        else:
            print(k, type(v).__name__)
print(edited())
```
