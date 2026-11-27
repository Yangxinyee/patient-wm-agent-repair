The latest fit recovers insulin dose changes fairly well, but still attenuates norepinephrine changes and leaves substantial factual residuals. That does not yet justify a payload edit. I’ll recover the architecture, causal estimates, and retained helpers before testing the complete transport path. Accurate dose contrasts with small factual perturbations would support it; attenuation or unstable responses would not.

```python
from pathlib import Path
import numpy as np, torch, inspect, re
torch.set_num_threads(8)

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA SCHEMA ===")
print(Path("/data/README.md").read_text())

notes188 = Path("/work/notes.md").read_text()
print("\n=== EARLY RECORD-IDENTIFICATION NOTES ===")
print(notes188[:16000])

print("\n=== RETAINED RECENT OBJECTS ===")
snap188 = list(globals().items())
for name, obj in snap188:
    if name.startswith("_") or not re.search(r"(154|169|171|179|180|185|186|187)$", name):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape), str(obj.dtype))
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), "keys:", list(obj.keys())[:20])
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj),
              [(type(x).__name__, getattr(x, "shape", None)) for x in obj[:5]])
    elif callable(obj):
        try:
            print(name, "function", inspect.signature(obj))
        except Exception:
            print(name, "function")
    elif isinstance(obj, (str, int, float, bool)):
        print(name, repr(obj)[:160])

print("\n=== AVAILABLE LOCAL FUNCTIONS ===")
for name, obj in snap188:
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            print(name, inspect.signature(obj))
        except Exception:
            print(name)
print("\nCurrent edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 188; forwards 10. Restart recovery: low-penalty latent decoding preserves insulin contrasts but NE stopping/high scales remain attenuated, and factual residuals are not negligible. Recovering architecture, original record-based identification, and retained replay/calibration objects before installing a payload. Existing five zero-payload transport edits unchanged.\n")
```
