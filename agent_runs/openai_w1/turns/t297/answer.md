The last run confirmed the physical-dose array, but its output was cut off before showing the recurrent-input features. The current edit is factually safe but has not materially improved dose responses. I’ll recover the direct-injection setup next; it is promising only if it preserves dose contrasts while leaving recorded trajectories nearly unchanged.

```python
from pathlib import Path
import inspect
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text()[:6900])

print("\n=== RELEVANT OBJECTS (SHAPES ONLY) ===")
keys261 = sorted(k for k in globals()
                 if k.startswith(("ds_", "lc_", "rp_"))
                 or k.endswith(("257", "258", "259", "260"))
                 or k in ("safe236", "cf_tokens259", "stays248"))
for k in keys261:
    v = globals()[k]
    if isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, type(v).__name__, tuple(v.shape))
    elif callable(v):
        c = getattr(v, "__code__", None)
        print(k, "function", c.co_varnames[:c.co_argcount] if c else "")
    elif isinstance(v, dict):
        print(k, "dict keys:", list(v)[:24])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, "length", len(v))
    elif np.isscalar(v):
        print(k, repr(v)[:100])

print("\n=== DIRECT INJECTION FUNCTIONS ===")
for k in sorted(globals()):
    if callable(globals()[k]) and k.endswith(("257", "258", "259", "260")):
        try:
            src = inspect.getsource(globals()[k])
        except (OSError, TypeError):
            src = ""
        if src:
            print(k, src[:1800])

if "B257" in globals():
    b261 = np.asarray(B257)
    print("B257", b261.shape, "column RMS",
          np.sqrt(np.mean(b261*b261, axis=0)))
print("core.b_in shape", SHAPES.get("core.b_in.weight"),
      SHAPES.get("core.b_in.bias"))
print("Current edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 261\n"
            "Run260 confirmed ds_rates physical-dose alignment but broad function printing "
            "truncated useful output. Recovering architecture and compact recurrent-input "
            "objects before a factual-anchored core.b_in edit. No weights changed. "
            "Counts: 261 Python runs, 27 forwards.\n")
```