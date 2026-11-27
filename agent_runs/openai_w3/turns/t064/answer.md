The pilot preserved factual NLL but barely changed any dose response. It therefore has not repaired the model. I’ll inspect the weight changes and correction scaling to distinguish an inactive correction from one that is simply too weak. A useful mechanism must produce a clear latent change under edited doses, not just preserve factual fit.

```python
import numpy as np
from pathlib import Path

pilot_weights_53 = {name: load(name).copy() for name in edited()}

print("EDIT MAGNITUDES")
for name in edited():
    original = load_original(name)
    current = load(name)
    difference = current - original
    print(name, current.shape,
          "max_abs", float(np.max(np.abs(difference))),
          "rms", float(np.sqrt(np.mean(difference ** 2))))
    if difference.ndim == 2:
        row_norms = np.linalg.norm(difference, axis=1)
        column_norms = np.linalg.norm(difference, axis=0)
        print("largest changed rows:",
              [(int(i), float(row_norms[i]))
               for i in np.argsort(row_norms)[-6:][::-1]])
        print("largest changed columns:",
              [(int(i), float(column_norms[i]))
               for i in np.argsort(column_norms)[-6:][::-1]])

print("\nPILOT-RELATED LIVE OBJECTS")
_snapshot57 = dict(globals())
for name, obj in sorted(_snapshot57.items()):
    if any(s in name.lower() for s in
           ("repair", "pilot", "strength", "selected", "post_hidden",
            "posterior_input", "policy_w", "policy_b", "gate_row", "gate_bias")):
        if hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))
            if np.size(obj) <= 12:
                print(np.asarray(obj))
        elif isinstance(obj, dict):
            print(name, "keys:", list(obj.keys())[:30])
        elif isinstance(obj, (list, tuple, int, float, str)):
            print(name, repr(obj)[:500])
        elif callable(obj):
            try:
                print(name, inspect.signature(obj))
            except Exception:
                print(name, "callable")

print("\nREMAINING LIVE NAMES")
print(" ".join(sorted(k for k in _snapshot57
                      if not k.startswith("_") and k >= "r40")))

print("\nEARLY NOTEBOOK: RECORD-DERIVED EVIDENCE")
print(Path("/work/notes.md").read_text()[:5000])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 57; forwards 5. Pilot validated in forwards 4/5: "
            "factual NLL difference +2.13e-8 nats/reading. Reached-row shifts "
            "original -> pilot: insulin x2 -0.422 -> -0.406, half +0.213 -> +0.199; "
            "NE x2 -0.742 -> -0.726, half -0.238 -> -0.230. "
            "Pilot is ineffective despite preserving factual fit. "
            "Saved pilot_weights_53 for restoration; diagnosing scaling and gating.\n")
```