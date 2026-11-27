The architecture confirms that doses affect the recurrent state, while generated readings feed back into later predictions. That makes it important to separate dose sensitivity from history sensitivity. The inspection output was truncated by a large default argument; I’ll repeat it compactly so we can validate the full transition before fitting any new edit.

```python
import inspect, numpy as np, pandas as pd, torch
from pathlib import Path

print("=== LOCAL HELPERS (NO DEFAULT VALUES) ===")
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            params = ", ".join(inspect.signature(obj).parameters)
        except Exception:
            params = "?"
        print(name + "(" + params + ")")

print("=== ARRAY / TABLE INVENTORY ===")
inventory = []
for name, obj in list(globals().items()):
    if name.startswith("_"):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor, pd.DataFrame)):
        inventory.append((name, type(obj).__name__, tuple(obj.shape)))
print("\n".join(str(x) for x in inventory))

print("=== SMALL CONFIGURATION OBJECTS ===")
terms = ("spec", "post", "state", "trans", "graft", "decoder", "gate",
         "beta", "rd_", "panel", "bundle", "probe", "checkpoint", "weight")
for name, obj in list(globals().items()):
    if name.startswith("_") or not any(t in name.lower() for t in terms):
        continue
    if isinstance(obj, dict):
        print(name, "keys:", str(list(obj.keys())[:16])[:260])
    elif isinstance(obj, (str, int, float, bool, tuple)):
        print(name, str(obj)[:220])
    elif isinstance(obj, list):
        print(name, "list length:", len(obj))

print("=== EARLY IDENTIFICATION NOTES ===")
notes_text = Path("/work/notes.md").read_text()
print(notes_text[:4200])
```