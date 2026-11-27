The architecture confirms that doses enter through both numeric features and shared dose/rate bins, followed by a nonlinear action encoder. The numeric edit tested only one of those routes. The context dump was truncated, so I’ll first recover the existing helper functions and diagnostics rather than recreate them. Those should reveal whether insulin’s numeric signal is being suppressed.

```python
import inspect
from pathlib import Path

inventory = []
for key, val in list(globals().items()):
    if key.startswith("_") or key in {"f", "val", "notes_text", "inventory"}:
        continue
    shape = getattr(val, "shape", None)
    if shape is not None:
        info = str(tuple(shape))
    elif isinstance(val, dict):
        info = "keys=" + ",".join(map(str, list(val)[:8]))
    elif isinstance(val, (list, tuple)):
        info = "len=" + str(len(val))
    elif inspect.isfunction(val):
        try:
            info = str(inspect.signature(val))
        except Exception:
            info = ""
    else:
        info = ""
    inventory.append(f"{key}: {type(val).__name__} {info}")
print("=== INVENTORY ===")
print("\n".join(inventory))
print("=== NOTEBOOK SECTION HEADINGS ===")
text = Path("/work/notes.md").read_text()
print("\n".join(line for line in text.splitlines() if line.startswith("#")))
print("=== CURRENT EDITS ===")
for name in edited():
    delta = np.asarray(load(name)) - np.asarray(load_original(name))
    print(name, tuple(delta.shape), "norm", float(np.linalg.norm(delta)),
          "max_abs", float(np.abs(delta).max()))
```
