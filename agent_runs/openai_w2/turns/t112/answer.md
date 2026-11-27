The unit search explains much of the reconstruction error: the correct measurement units reproduce single-reading states almost exactly. I’ll recover the architecture and existing helpers, then check complete bundles. Close agreement across bundle types would support a targeted edit; remaining systematic errors would mean reconstruction still needs fixing.

```python
import inspect
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== EARLIER NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:6500])

print("\n=== EXISTING FUNCTIONS ===")
for _recover_name, _recover_obj in sorted(list(globals().items())):
    if inspect.isfunction(_recover_obj) and getattr(_recover_obj, "__module__", "") == "__main__":
        try:
            print(_recover_name, inspect.signature(_recover_obj))
        except Exception:
            print(_recover_name)

print("\n=== DATA AND FEATURE OBJECTS ===")
for _recover_name, _recover_obj in sorted(list(globals().items())):
    if _recover_name.startswith("_"):
        continue
    if isinstance(_recover_obj, np.ndarray):
        print(_recover_name, "array", _recover_obj.shape, _recover_obj.dtype)
    elif isinstance(_recover_obj, pd.DataFrame):
        print(_recover_name, "DataFrame", _recover_obj.shape,
              list(_recover_obj.columns)[:24])
    elif isinstance(_recover_obj, dict):
        print(_recover_name, "dict", len(_recover_obj),
              list(_recover_obj.keys())[:12])
    elif isinstance(_recover_obj, (list, tuple)) and len(_recover_obj):
        print(_recover_name, type(_recover_obj).__name__, len(_recover_obj))

print("\nCurrent edits:", edited())
```