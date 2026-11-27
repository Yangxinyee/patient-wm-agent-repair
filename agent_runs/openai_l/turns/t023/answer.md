```python
from pathlib import Path
import json, inspect, numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== VOCAB STRUCTURE ===")
print({k: (type(v).__name__, len(v) if hasattr(v, "__len__") else None)
       for k, v in VOCAB.items()})
for k, v in VOCAB.items():
    if any(s in k.lower() for s in ["numeric", "norm", "code", "size"]):
        print(k, json.dumps(v, default=str)[:14000])

print("\n=== PERSISTENT ANALYSIS OBJECTS ===")
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(v, np.ndarray):
        print(k, "array", v.shape, v.dtype)
    elif callable(v) and getattr(v, "__module__", "") == "__main__":
        try:
            print(k, "function", inspect.signature(v))
        except Exception:
            print(k, "function")
    elif isinstance(v, dict) and k not in {"__builtins__"}:
        print(k, "dict keys", list(v.keys())[:18])
    elif hasattr(v, "shape") and hasattr(v, "columns"):
        print(k, "table", v.shape, "columns", list(v.columns)[:15])

print("\n=== FORWARD 3 FILES AND SUMMARY ===")
p20 = Path("/work/fwd_3")
print([p.name for p in p20.iterdir()])
if (p20 / "summary.json").exists():
    print((p20 / "summary.json").read_text())

print("\n=== EARLIER NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:8500])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 20 / restart recovery\n"
            "Cross-sensor impulse responses support a delayed insulin effect and faster-decaying NE effect. "
            "AR1 is not accepted; inspecting the exact architecture and retained analysis objects before any weight edit. "
            "No tensors edited; forward passes remain 3/50.\n")
```
