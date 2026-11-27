The calibration barely helps on the held-out cohort, so it does not make the combined edit safe. I’ll recover the exact edit and cache construction before revising it. A useful revision must retain dose sensitivity while reducing factual loss below the limit on independent stays.

```python
from pathlib import Path
import inspect, numpy as np, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
print("=== RECENT NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[-18000:])
print("=== RECENT KERNEL VARIABLES ===")
for name in sorted(list(globals())):
    if name.startswith(("r170", "r171", "r176", "r178", "r183", "c183", "c185",
                        "c196", "r198", "r199", "r200", "r201", "r202", "r203",
                        "r204", "r205", "r206", "r207", "r208", "r209", "r210", "r211")):
        v = globals()[name]
        if callable(v):
            try:
                print(name, inspect.signature(v))
            except Exception:
                print(name, type(v).__name__)
        elif isinstance(v, dict):
            print(name, "dict keys:", list(v)[:35])
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            print(name, type(v).__name__, tuple(v.shape), v.dtype)
        elif hasattr(v, "shape"):
            print(name, type(v).__name__, v.shape)
        else:
            print(name, type(v).__name__, repr(v)[:250])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 212: restart and calibration audit\n"
            "Moment calibration does not rescue combined factual loss on tuning 100–199. "
            "Recovering the exact edit and independent caches before a constrained revision. "
            "No weights changed.\n")
```