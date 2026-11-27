The reconstruction matches isolated drug starts, but not complete sequences, so it is not yet reliable enough for a weight edit. I’ll review the architecture, retained causal estimates, and kernel variables to locate the mismatch. Matching all bundle types would support proceeding; systematic errors would require fixing the reconstruction first.

```python
import os, inspect, numpy as np
print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("\n=== DATA DEFINITIONS ===")
print(open("/data/README.md").read())
print("\n=== EARLIER NOTEBOOK ===")
notes_text = open("/work/notes.md").read()
cut = notes_text.find("## Run 10:")
print(notes_text[:cut if cut >= 0 else 14000])
print("\n=== RETAINED VARIABLES ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or name in {"NAMES", "SHAPES", "VOCAB", "notes_text"}:
        continue
    if isinstance(obj, np.ndarray):
        print(name, "array", obj.shape, obj.dtype)
    elif hasattr(obj, "shape") and not inspect.ismodule(obj):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif inspect.isfunction(obj):
        print(name, "function", str(inspect.signature(obj)))
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), "keys:", list(obj)[:12])
    elif isinstance(obj, (list, tuple)) and len(obj) > 10:
        print(name, type(obj).__name__, len(obj), "first:", str(obj[:3])[:180])
    elif isinstance(obj, (str, int, float, list, tuple, bool)):
        print(name, str(obj)[:220])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 20: restart recovery\n"
            "Full posterior reconstruction still disagrees on reading bundles and some END bundles; "
            "recovering architecture, causal evidence, and retained objects before any edit. "
            "No tensors edited. Python runs: 20. Forward passes: 3.\n")
```
