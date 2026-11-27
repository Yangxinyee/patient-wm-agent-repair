The notebook index locates the threshold tests and sustained-effect calculation, but the previous output was truncated before showing them. I’ll read just that section now. Agreement across threshold bandwidths and outcome horizons would support using its insulin estimate; disagreement would require revisiting the target.

```python
print("=== THRESHOLD IDENTIFICATION AND SUSTAINED TARGET ===")
print(notes_text[23378:30112])

print("\n=== RD DYNAMIC PARAMETERS ===")
print(rd_dynamic_fit.get("x"), "cost", rd_dynamic_fit.get("cost"))

print("\n=== RECONSTRUCTION ARRAY INVENTORY ===")
for name, obj in list(globals().items()):
    if not name.startswith(("rec_", "rm_", "r131_")) or inspect.isfunction(obj):
        continue
    if isinstance(obj, dict):
        print(name, "dict", list(obj.keys())[:4])
    elif hasattr(obj, "shape") and not callable(getattr(obj, "shape", None)):
        print(name, tuple(obj.shape))
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj))
```
