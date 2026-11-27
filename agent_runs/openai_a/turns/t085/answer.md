The graft has measurable but modest leverage: insulin doubling now gives −2.375 rather than −1.847, while halving gives +1.062 rather than +0.777. Norepinephrine is nearly unchanged. I’ll inspect how the graft is scaled before increasing it; a stronger version must retain factual fit and move both insulin readouts toward the record-based estimates.

```python
import inspect, json
from pathlib import Path

with open("/work/notes.md", "a") as f:
    f.write("""
## GPU validation of latent-robust graft; Python 73, forwards 8–9
- beta=-2 graft passed paired factual check on first 100 tuning stays:
  original NLL 1.63461, edited 1.63453 (delta -0.00008).
- Paired rollouts: insulin x2 -1.847 -> -2.375; half .777 -> 1.062.
  NE x2 -.498 -> -.505; half -.600 -> -.605.
- Insulin leverage is real but too weak relative to the earlier RD-derived target.
- Inspecting the retained graft construction before increasing its strength.
""")

print("FUNCTIONS")
for name in sorted(list(globals())):
    obj = globals()[name]
    if callable(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            print(name, str(inspect.signature(obj)))
        except Exception:
            print(name)

print("\nGRAFT / ROBUST / BETA VARIABLES")
for name in sorted(list(globals())):
    if any(s in name.lower() for s in ["graft", "robust", "beta", "pool3", "decoder3", "gate3"]):
        obj = globals()[name]
        if isinstance(obj, np.ndarray):
            print(name, "array", obj.shape, "norm", float(np.linalg.norm(obj)))
        elif isinstance(obj, dict):
            print(name, "dict", list(obj.keys())[:20])
        elif isinstance(obj, (str, int, float, bool, list, tuple)):
            print(name, repr(obj)[:500])

print("\nNOTEBOOK IDENTIFICATION EVIDENCE")
notes_text = Path("/work/notes.md").read_text()
print(notes_text[:6500])
```
