The unsuccessful probe is now fully reverted. Before trying another gain, I’ll recover the calibration directions and the original record-based evidence so I can distinguish a sign error from a weak pathway. A useful direction must show the expected response in actual rollouts; an approximate local reconstruction is insufficient.

```python
import inspect, numpy as np
from pathlib import Path

print("=== CALIBRATION AND EDIT OBJECTS ===")
for key, value in sorted(list(globals().items())):
    if key.startswith("_") or not key.endswith(("161", "166", "167", "168", "169")):
        continue
    if inspect.ismodule(value):
        continue
    if callable(value):
        try:
            print(key, inspect.signature(value))
        except Exception:
            print(key, type(value).__name__)
    elif isinstance(value, dict):
        print(key, "keys:", list(value.keys())[:25])
    elif hasattr(value, "shape") and not callable(value.shape):
        ar = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        print(key, ar.shape, "rms", float(np.sqrt(np.mean(ar.astype(float)**2))),
              "range", (float(np.nanmin(ar)), float(np.nanmax(ar))))
    elif isinstance(value, (tuple, list)):
        print(key, repr(value)[:300] if len(value) < 15 else "length=" + str(len(value)))
    else:
        print(key, repr(value)[:200])

print("\n=== RETAINED FUNCTIONS ===")
for key, value in sorted(list(globals().items())):
    if inspect.isfunction(value) and not key.startswith("_"):
        try:
            print(key, inspect.signature(value))
        except Exception:
            pass

print("\n=== EARLY RECORD EVIDENCE ===")
notes171 = Path("/work/notes.md").read_text()
print(notes171[:5000])

print("\n=== PROBE WEIGHT MAGNITUDES ===")
for key in candidate_probe170:
    old = np.asarray(load_original(key))
    new = np.asarray(candidate_probe170[key])
    print(key, "original rms", np.sqrt(np.mean(old**2)),
          "edit rms", np.sqrt(np.mean((new-old)**2)),
          "max change", np.max(np.abs(new-old)))
```