The probe produced states for all 100 stays. Those are diagnostic inputs, not evidence that the repair works. I’ll now restore the original weights and recover the architecture and prior effect estimates, so the next edit targets a pathway the model actually uses.

```python
import os, json, numpy as np, pandas as pd, torch
_live68 = list(globals().items())
probe7 = np.load("/work/fwd_7/states_edited.npz")
for name in list(edited()):
    drop_edit(name)
print("Restored original; edits:", edited())
print("PROBE ARRAYS:", {k: probe7[k].shape for k in probe7.files})
print("\nARCHITECTURE\n", open("/model/ARCHITECTURE.md").read())
print("\nDATA SCHEMA\n", open("/data/README.md").read())
print("\nLIVE FUNCTIONS")
print([(k, getattr(v, "__module__", "")) for k, v in _live68
       if callable(v) and getattr(v, "__module__", "") == "__main__"])
print("\nLIVE TABLES/ARRAYS/DICTIONARIES")
for k, v in _live68:
    if k.startswith("_"):
        continue
    if isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict) and k not in ("SHAPES", "VOCAB"):
        print(k, "dict", len(v), "keys:", list(v)[:12])
print("\nEARLY NOTEBOOK\n", open("/work/notes.md").read()[:11000])
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 68; forwards 7. Probe states acquired for 100 training stays. All four diagnostic posterior edits immediately dropped; original model restored. Recovering architecture, statistical evidence and live helpers before choosing an effective correction path.\n")
```