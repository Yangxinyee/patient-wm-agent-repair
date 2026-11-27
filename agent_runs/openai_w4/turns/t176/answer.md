```python
from pathlib import Path
import inspect, numpy as np, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== DATA COLUMNS ===")
print(Path("/data/README.md").read_text()[:6000])

print("\n=== RETAINED REPAIR / EFFECT OBJECTS ===")
keys156 = sorted(k for k in globals() if any(s in k.lower() for s in
    ["chosen_channel", "direction", "lp_new", "kernel", "target", "replay", "transition", "innovation", "causal", "effect", "scale_results", "nll_scales"]))
for k in keys156:
    v = globals()[k]
    if callable(v):
        try: desc = str(inspect.signature(v))
        except Exception: desc = "callable"
    elif isinstance(v, dict):
        desc = "dict keys=" + str(list(v.keys())[:20])
    elif isinstance(v, (np.ndarray, torch.Tensor)):
        desc = "shape=" + str(tuple(v.shape))
    else:
        desc = str(type(v).__name__)
    print(k, desc)

print("\n=== CHANNEL PARAMETERS ===")
for k in ["chosen_channel116", "directions118", "lp_new136"]:
    v = globals().get(k)
    if isinstance(v, dict):
        for kk, vv in v.items():
            if isinstance(vv, (np.ndarray, torch.Tensor)):
                a = np.asarray(vv)
                print(k, kk, a.shape, "first=", a.reshape(-1)[:14])
            elif isinstance(vv, dict):
                print(k, kk, "keys=", list(vv)[:18])
            else:
                print(k, kk, str(vv)[:350])

notes156 = Path("/work/notes.md").read_text()
print("\n=== EARLY IDENTIFICATION NOTES ===")
parts156 = notes156.split("\n## ")
for p in parts156:
    if any(s in p.lower() for s in ["identification", "causal", "lag", "innovation", "channel", "bootstrap"]):
        if not any(p.startswith(f"Run {j}") for j in range(123, 156)):
            print(("## " + p)[:850])
Path("/work/notes.md").open("a").write(
    "\n## Run 156\nRecovered architecture and retained identification/channel objects. "
    "Run-155 insulin dose-channel slopes are near proportional; NE channel slopes are weak and inconsistent. "
    "This diagnoses encoding only, not drug effects. No weights changed.\n")
```
