The export succeeded and gives us the current model’s actual bundle states. It does not yet test causal response or factual safety. I’ll recover the reconstruction code and align these states with the records; matching bundle identities and local replay will support fitting a new projection, while mismatches would invalidate it.

```python
from pathlib import Path
import inspect, re
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

with np.load("/work/fwd_30/states_edited.npz") as f:
    S280 = {k: f[k].copy() for k in f.files}
print("\n=== CURRENT STATE EXPORT ===")
for k, v in S280.items():
    print(k, v.shape, v.dtype)
print("Unique stays:", len(np.unique(S280["stay_id"])))

g280 = dict(globals())
print("\n=== RECENT RECONSTRUCTION OBJECTS ===")
for name in sorted(g280):
    if not re.search(r"27[0-9]|269|safe236|ds_rates|factualE258|Es259", name):
        continue
    obj = g280[name]
    if callable(obj):
        try:
            desc = str(inspect.signature(obj))
        except Exception:
            desc = "<callable>"
    elif hasattr(obj, "shape"):
        desc = f"shape={tuple(obj.shape)}, dtype={getattr(obj, 'dtype', '?')}"
    elif isinstance(obj, dict):
        desc = f"dict[{len(obj)}], keys={list(obj)[:12]}"
    elif isinstance(obj, (list, tuple)):
        desc = f"{type(obj).__name__}[{len(obj)}]"
    elif isinstance(obj, str):
        desc = f"str[{len(obj)}]"
    else:
        desc = repr(obj)[:180]
    print(name, desc)

print("\n=== AVAILABLE LOCAL FUNCTIONS ===")
for name, obj in sorted(g280.items()):
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = ""
        print(name, sig)

notes280 = Path("/work/notes.md").read_text()
print("\n=== EARLIEST RECORD-IDENTIFICATION NOTES ===")
print(notes280[:6500])
print("\nCurrent edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 280\nForward30 exported 5215 actual safe236 bundle states on the aligned 100 training stays. Recovering architecture and reconstruction functions before fitting corrected factual-anchored features. No weights changed. Counts: 280 Python runs, 30 forwards.\n")
```
