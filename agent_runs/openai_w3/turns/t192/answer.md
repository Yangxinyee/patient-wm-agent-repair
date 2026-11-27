The last calibration required large weight changes but still reconstructed the intended effects poorly, so it is not safe to install. I will recover the architecture and saved calibration objects to check where the dose signal is being lost. Accurate transport of a known signal with small factual changes would support this approach; another reconstruction failure would refute it.

```python
import os, json, inspect, numpy as np, torch
print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("\n=== DATA SCHEMA ===")
print(open("/data/README.md").read())
print("\n=== EARLY NOTEBOOK ===")
notes172 = open("/work/notes.md").read()
print(notes172[:18000])
print("\n=== RETAINED OBJECTS (RECENT / CALIBRATION) ===")
for k in sorted(list(globals())):
    if k.startswith("_"):
        continue
    v = globals()[k]
    recent = any(k.endswith(str(j)) for j in range(149, 172))
    relevant = any(s in k.lower() for s in ["jac", "direction", "transport", "replay", "posterior", "policy", "decode", "payload", "reference"])
    if recent or relevant:
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, type(v).__name__, tuple(v.shape), str(v.dtype))
        elif isinstance(v, dict):
            print(k, "dict", list(v.keys())[:25])
        elif callable(v):
            print(k, "callable", str(inspect.signature(v)))
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, "len", len(v))
        elif isinstance(v, (str, int, float, bool)):
            print(k, repr(v)[:250])
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 172; forwards 10. Run 171 calibration was ill-conditioned and failed output-space reconstruction; no additional weights installed. Recovering architecture and retained objects before validating a unit payload transport and recalibrating physiology. Five zero-payload transport tensors remain edited.\n")
```