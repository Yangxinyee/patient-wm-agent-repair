The last pass collected paired factual states; it did not measure treatment effects. I’ll inspect how the numeric edit changes those states and recover the model’s update equations. A useful repair direction should distinguish dose-driven changes from ordinary factual variation; substantial overlap would argue against amplifying it.

```python
import os, json, numpy as np
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== PERSISTENT VARIABLE INVENTORY ===")
for name, value in sorted(list(globals().items())):
    if name.startswith("_") or name in {"In", "Out", "get_ipython"}:
        continue
    if isinstance(value, np.ndarray):
        print(name, "array", value.shape, value.dtype)
    elif isinstance(value, dict):
        print(name, "dict", len(value), "keys:", list(value)[:12])
    elif isinstance(value, (list, tuple)):
        print(name, type(value).__name__, len(value))
    elif callable(value) and getattr(value, "__module__", "") == "__main__":
        print(name, "function")

print("\n=== PAIRED STATE FILES ===")
for label in ["original", "edited"]:
    archive = np.load(f"/work/fwd_24/states_{label}.npz")
    print(label, {key: (archive[key].shape, str(archive[key].dtype))
                  for key in archive.files})
    if label == "original":
        state24_original = {key: archive[key].copy() for key in archive.files}
    else:
        state24_edited = {key: archive[key].copy() for key in archive.files}

assert np.array_equal(state24_original["stay_id"], state24_edited["stay_id"])
assert np.array_equal(state24_original["hour"], state24_edited["hour"])
delta24 = state24_edited["x"] - state24_original["x"]
for label, sl in [
    ("out", slice(0, 192)),
    ("h", slice(192, 960)),
    ("z", slice(960, 992)),
]:
    dx = delta24[:, sl]
    xx = state24_original["x"][:, sl]
    print(label, "delta RMS", float(np.sqrt(np.mean(dx**2))),
          "centered factual RMS", float(np.sqrt(np.mean((xx-xx.mean(0))**2))))

print("\n=== CURRENT OPERATIONS ===")
print(ops())
with open("/work/notes.md", "a") as f:
    f.write("\n## Restart / Python run 55\n"
            "- Forward 24 collected original and numeric-gain-4 factual states on 100 tuning stays.\n"
            "- Recovering architecture and comparing paired states before designing another selective edit.\n"
            "- No edits changed. Forward passes used: 24.\n")
```