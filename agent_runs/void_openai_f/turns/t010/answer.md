The last run confirmed that no weights have changed and that the cross-fitted analysis is still missing. We therefore have no validated repair yet. I’ll recover the architecture and saved regression results before choosing an edit. Stable dose effects across history-adjusted analyses would support a causal interpretation; sensitivity to those adjustments would weaken it.

```python
import os, json, numpy as np, pandas as pd, torch

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())

print("\n=== DATA README ===")
print(open("/data/README.md").read())

print("\n=== RETAINED VARIABLES ===")
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(v, (pd.DataFrame, pd.Series, np.ndarray, dict, list, tuple)):
        shape = getattr(v, "shape", None)
        print(k, type(v).__name__, shape if shape is not None else f"len={len(v)}")

print("\n=== SAVED ESTIMATES ===")
if "estimates" in globals():
    print(repr(estimates)[:18000])

print("\n=== VOCAB STRUCTURE ===")
print("Top-level keys:", list(VOCAB))
for k, v in VOCAB.items():
    if not isinstance(v, dict):
        print(k, repr(v)[:1200])
    else:
        print(k, "keys:", list(v)[:40])

print("\n=== FORWARD ARTIFACTS ===")
for d in sorted(os.listdir("/work")):
    if d.startswith("fwd_") and os.path.isdir("/work/" + d):
        print(d, os.listdir("/work/" + d))
        for f in os.listdir("/work/" + d):
            p = "/work/" + d + "/" + f
            if f.endswith((".json", ".txt", ".log")) and os.path.getsize(p) < 30000:
                print(f, open(p).read()[:3000])

with open("/work/notes.md", "a") as f:
    f.write("\n## Restart recovery\nNo edits submitted. Recovering architecture, retained estimates, and forward-error artifacts before further inference. Python runs: 6; successful forward passes: 0.\n")
```