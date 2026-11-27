```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RELEVANT HELPERS / OBJECTS (compact) ===")
selected = []
for k, v in list(globals().items()):
    if k.startswith("_"):
        continue
    if any(s in k.lower() for s in [
        "transport", "compens", "install", "checkpoint", "snapshot",
        "candidate206", "restricted199", "response186", "view171",
        "phys171", "prelude", "replay", "214"
    ]):
        selected.append((k, v))
for k, v in sorted(selected):
    if callable(v):
        try:
            print(k, "function", inspect.signature(v))
        except Exception:
            print(k, "callable")
    elif isinstance(v, dict):
        print(k, "dict", list(v.keys())[:24])
    elif hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, str):
        print(k, "string length", len(v))
    else:
        print(k, type(v).__name__)

for k in ["response186", "view171", "phys171"]:
    v = globals().get(k)
    if callable(v):
        print("\n=== SOURCE", k, "===")
        try:
            print(inspect.getsource(v))
        except Exception as e:
            print("Source unavailable:", type(e).__name__)

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 215; forwards 14. Insulin carrier and attention contrasts are bidirectional; rollout asymmetry appears downstream, not a simple dose-decoding failure. Recovering recurrence and transport helpers before testing generated-history feedback. No weights changed.\n")
```