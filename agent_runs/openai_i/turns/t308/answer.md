```python
from pathlib import Path
import numpy as np, inspect, dis

arch271 = Path("/model/ARCHITECTURE.md").read_text()
print("=== ARCHITECTURE ===")
print(arch271[:6200])
print("\n=== CANDIDATE DIFFERENCES ===")
c271 = {k: np.array(v) for k, v in np.load("/work/corrected269.npz").items()}
n271 = {k: np.array(v) for k, v in np.load("/work/neutral209.npz").items()}
for k in c271:
    base = n271[k] if k in n271 else np.asarray(load_original(k))
    d = c271[k] - base
    print(k, c271[k].shape, "delta_norm", round(float(np.linalg.norm(d)), 6),
          "changed_rows", int(np.sum(np.any(abs(d) > 1e-7, axis=1))) if d.ndim == 2 else int(np.sum(abs(d) > 1e-7)))
print("\n=== RETAINED ROUTING HELPERS ===")
for k, v in list(globals().items()):
    if callable(v) and hasattr(v, "__code__") and (
        any(s in k.lower() for s in ["response", "route", "install", "replay"])
        or k.endswith(("267", "268", "269"))
    ):
        print(k, inspect.signature(v), "names", v.__code__.co_names)
print("\n=== RECENT FORWARD FILES ===")
for p in sorted(Path("/work").glob("fwd_*/factual_*.csv"))[-8:]:
    print(str(p), p.stat().st_size)
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 271 — corrected-route refinement audit\n"
            "- corrected269 remains provisional: factual dNLL +.00546 exceeds the .005 limit.\n"
            "- Inspecting exact routing changes before reducing their footprint; keeping the empty-action correction.\n"
            "- No weights changed in this run.\n")
```