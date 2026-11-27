The encoding checks reproduce the saved posterior states almost exactly. That supports reconstructing local steps, but does not establish drug effects. I’ll now recover the transition details and retained helpers. Matching complete GPU state updates would justify synthetic-dose tests; a mismatch would rule them out.

```python
from pathlib import Path
import inspect, numpy as np, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("=== RETAINED FUNCTIONS ===")
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and not name.startswith("_"):
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "(signature unavailable)"
        print(name, sig)

print("=== RECONSTRUCTION / GRAFT VARIABLES ===")
terms = ("spec", "post", "state", "trans", "graft", "decoder", "ridge",
         "gate", "beta", "rd_", "panel", "bundle", "probe", "checkpoint")
for name, obj in list(globals().items()):
    if name.startswith("_") or not any(t in name.lower() for t in terms):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict", list(obj.keys())[:20])
    elif isinstance(obj, (str, int, float, bool, tuple)):
        print(name, repr(obj)[:350])
    elif isinstance(obj, list):
        print(name, "list", len(obj), repr(obj[:2])[:250])

print("=== CURRENT EDITS ===", edited())
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Restart after Python 118; forward 11/50\n"
            "- Insulin posterior reconstruction RMSE ~2e-7; measurement posterior ~1.7e-6.\n"
            "- Retaining validated beta=-2 graft while recovering transition helpers.\n"
            "- These are encoding validations, not new evidence about causal effects.\n")
```