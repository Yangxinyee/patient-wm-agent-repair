The reconstruction is accurate for bundles without lab glucose; the remaining error is isolated to bundles containing `glucose_lab`. Its different encoding may explain that. I’ll review the retained helpers and earlier record-based estimates, then verify the correction before using reconstructed states to edit weights. Persistent bundle errors would invalidate that approach.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== COMPLETE NOTEBOOK ===")
print(Path("/work/notes.md").read_text())

print("\n=== RETAINED VARIABLES AND HELPERS ===")
r20_snapshot = list(globals().items())
for r20_name, r20_obj in r20_snapshot:
    if r20_name.startswith("_") or r20_name.startswith("r20_"):
        continue
    if inspect.isfunction(r20_obj) and getattr(r20_obj, "__module__", "") == "__main__":
        try:
            print("FUNCTION", r20_name, inspect.signature(r20_obj),
                  "references:", r20_obj.__code__.co_names)
        except Exception:
            print("FUNCTION", r20_name)
    elif hasattr(r20_obj, "shape"):
        print("DATA", r20_name, type(r20_obj).__name__, tuple(r20_obj.shape))
    elif isinstance(r20_obj, dict):
        print("DICT", r20_name, "length", len(r20_obj),
              "first keys", list(r20_obj.keys())[:8])
    elif isinstance(r20_obj, (list, tuple)):
        print("SEQUENCE", r20_name, "length", len(r20_obj))
    elif isinstance(r20_obj, (str, int, float, bool)):
        print("SCALAR", r20_name, str(r20_obj)[:160])

with Path("/work/notes.md").open("a") as r20_file:
    r20_file.write(
        "\n## Run 20: restart audit\n"
        "- Last reconstruction errors are isolated to bundles containing glucose_lab; "
        "single-lab validation identifies kq=5, unlike finger glucose kq=0.\n"
        "- Re-reading architecture, retained helpers, and earlier causal estimates "
        "before correcting full-bundle reconstruction.\n"
        "- Python runs: 20; forward passes: 4; no weights edited.\n"
    )
```