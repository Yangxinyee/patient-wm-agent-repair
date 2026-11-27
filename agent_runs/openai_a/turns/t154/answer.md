The recurrence matches the documented structure, but the memory tests so far added age information before the key/value projection. The architecture says age information goes to the keys, which may mean after projection. I’ll check the attention helper; a post-projection age test could explain the remaining state error.

```python
compact_dis(rec_attn_apply, 4700)
compact_dis(rec_attention, 2500)

print("\n=== BEST MEMORY TRIAL ===")
print(r131_best)

print("\n=== ATTENTION SIGNATURES ===")
print(inspect.signature(rec_attn_apply))
print(inspect.signature(rec_attention))

print("\n=== AVAILABLE CORRECTED-TRIAL OBJECTS ===")
for name, obj in list(globals().items()):
    if not name.startswith("r132_"):
        continue
    if hasattr(obj, "shape") and not callable(getattr(obj, "shape", None)):
        print(name, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, list(obj.keys())[:5])
    elif isinstance(obj, (tuple, list)):
        print(name, type(obj).__name__, len(obj))
```
