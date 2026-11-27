The architecture confirms that drug effects must pass through the recurrent state; a reading’s register does not directly contain drug doses. The output was cut off before the helper inventory, so I’ll retrieve that separately. This will let me test the action pathway reproducibly rather than reuse an uncertain reconstruction.

```python
import inspect
import types

print("=== REMAINING ENCODER / HEAD TENSORS ===")
for name in NAMES:
    if name.startswith(("encode", "z_", "last_", "query", "type_", "filter", "pending", "monitor", "embed.numeric")):
        print(name, SHAPES[name])

print("=== LOCAL HELPERS ===")
helper_lines100 = []
for name, obj in sorted(list(globals().items())):
    if isinstance(obj, types.FunctionType) and obj.__module__ == "__main__":
        try:
            helper_lines100.append(f"{name}{inspect.signature(obj)}")
        except Exception:
            helper_lines100.append(name)
print("\n".join(helper_lines100)[:4000])

print("=== DATA / CANDIDATE OBJECTS ===")
inventory100 = []
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or isinstance(obj, (types.ModuleType, types.FunctionType, type)):
        continue
    shape = getattr(obj, "shape", None)
    if shape is not None and not callable(shape):
        inventory100.append(f"{name}:{type(obj).__name__}{tuple(shape)}")
    elif isinstance(obj, (dict, list, tuple)):
        inventory100.append(f"{name}:{type(obj).__name__}[{len(obj)}]")
print("\n".join(inventory100)[:4500])
print("EDITS", edited())
```