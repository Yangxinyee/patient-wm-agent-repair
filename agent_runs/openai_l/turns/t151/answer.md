The last check showed independent control of the four private readout directions, but the controller equations disagreed. I’ll recover the architecture and fitted interfaces before changing more weights. Agreement with finite-perturbation tests would support the proposed correction; disagreement would rule it out.

```python
import os, json, inspect, numpy as np, torch
print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("=== DATA README ===")
print(open("/data/README.md").read())
_notes139 = open("/work/notes.md").read()
print("=== EARLY EVIDENCE ===")
print(_notes139[:18000])
print("=== NOTEBOOK SECTION INDEX ===")
print("\n".join(x for x in _notes139.splitlines() if x.startswith("#")))
print("=== RELEVANT GLOBALS ===")
for _n139, _v139 in list(globals().items()):
    if _n139.startswith("_"):
        continue
    if any(s in _n139.lower() for s in ["125", "118", "122", "signal", "cache", "controller", "wire", "kernel", "causal", "preamble", "recur"]):
        if isinstance(_v139, (np.ndarray, torch.Tensor)):
            print(_n139, type(_v139).__name__, tuple(_v139.shape),
                  "norm", float(np.linalg.norm(_v139.detach().cpu().numpy() if isinstance(_v139, torch.Tensor) else _v139)))
        elif callable(_v139):
            try:
                print(_n139, "callable", inspect.signature(_v139))
            except Exception:
                print(_n139, "callable")
        elif isinstance(_v139, dict):
            print(_n139, "dict keys", list(_v139)[:40])
        else:
            print(_n139, type(_v139).__name__, str(_v139)[:1200])
print("=== ALL CALLABLE INTERFACES ===")
for _n139, _v139 in list(globals().items()):
    if not _n139.startswith("_") and inspect.isfunction(_v139):
        try:
            print(_n139, inspect.signature(_v139))
        except Exception:
            pass
print("=== EDITED ===", edited())
```
