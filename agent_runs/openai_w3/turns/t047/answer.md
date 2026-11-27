The action features contain treatment information, but their held-out extraction errors are too large to trust for a weight edit. I’ll inspect the model components and live helpers to find a cleaner dose-sensitive path. It must represent changed doses accurately before we test factual fit and rollouts.

```python
from pathlib import Path
import numpy as np, torch, json, inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== LIVE VARIABLES ===")
for key in sorted(list(globals())):
    if key.startswith("_") or key in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    obj = globals()[key]
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(key, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(key, "dict", len(obj), "keys:", list(obj)[:8])
    elif callable(obj) and getattr(obj, "__module__", "") == "__main__":
        code = getattr(obj, "__code__", None)
        print(key, "function", code.co_varnames[:code.co_argcount] if code else "")
print("\n=== TENSOR NAMES AND SHAPES ===")
for name in NAMES:
    print(name, SHAPES[name])
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 42; forwards 3. Restart: action-rate extractors are not accurate enough for a safe edit. Inspecting architecture and live helpers to choose an exact dose-sensitive correction path. No weights edited.\n")
```