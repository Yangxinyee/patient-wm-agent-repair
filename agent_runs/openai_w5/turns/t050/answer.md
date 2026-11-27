The reconstructed recurrence is close but still differs from the saved states, so I would not yet trust an edit computed from it. I’ll inspect the update equations and existing helpers before choosing a targeted dose-pathway edit. The hypothesis will be supported only if the edit improves dose responses while keeping factual NLL within 0.005 nats of the original.

```python
from pathlib import Path
import inspect

print(Path("/model/ARCHITECTURE.md").read_text())
print("\n--- Persistent analysis objects ---")
for name, obj in list(globals().items()):
    if name.startswith("_") or name in {"In", "Out", "NAMES", "SHAPES", "VOCAB"}:
        continue
    if inspect.isfunction(obj):
        print(name, "function", str(inspect.signature(obj)))
    elif hasattr(obj, "shape"):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict", list(obj.keys())[:15])
    elif isinstance(obj, (list, tuple)) and len(obj) <= 30:
        print(name, type(obj).__name__, repr(obj)[:300])

print("\n--- Tensor names and shapes ---")
for name in NAMES:
    print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write("\n## Restart after run 41\n"
            "- Best local recurrence still has hidden-state RMSE about 3.74 versus slot SD 65.61; "
            "latent RMSE about 0.052. This is approximate, not validated exact reconstruction.\n"
            "- No current edits; 6 forward passes used. Reinspect equations and persistent helpers "
            "before selecting a targeted, empirically validated dose-pathway edit.\n")
```
