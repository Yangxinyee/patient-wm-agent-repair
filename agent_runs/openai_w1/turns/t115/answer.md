```python
import inspect, dis, pathlib, numpy as np

print("=== ARCHITECTURE ===")
print(pathlib.Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RECENT HELPERS AND ARRAYS ===")
for prefix in ("r88", "r89", "r100", "r101", "r102", "r103", "r104"):
    for name in sorted(n for n in globals() if n.startswith(prefix)):
        obj = globals()[name]
        if callable(obj):
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "?"
            code = getattr(obj, "__code__", None)
            print(name, sig, "dependencies:", code.co_names if code else None)
        elif isinstance(obj, np.ndarray):
            print(name, "array", obj.shape, obj.dtype)
        elif isinstance(obj, dict):
            print(name, "dict", len(obj), "keys:", list(obj)[:12])
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, len(obj), repr(obj[:3])[:350])
        else:
            print(name, type(obj).__name__, repr(obj)[:200])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 105\nDose-only reconstruction is validated, but the fresh observation path is not. Inspecting architecture and retained exact helpers before any calibration or weight edit. No edits. Counts: 105 Python runs, 6 forward passes.\n")
```