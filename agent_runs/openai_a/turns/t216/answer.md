The architecture confirms that the current edit changes only the value head, not the recurrent state update. A finger-selective gate remains plausible, but I need the exact graft construction and causal estimates before changing it. I’ll recover those next; the gate must preserve MAP predictions and suppress the lab correction without weakening the finger correction.

```python
import inspect, numpy as np, torch, re

print("CAUSAL EVIDENCE FROM NOTEBOOK")
sections = re.split(r"(?m)(?=^## )", notes_text)
wanted = [
    "Original rollout and identification",
    "Factual baseline and dynamic identification",
    "Sensor-contrast local projections",
    "Dynamic identification;",
    "Recovery and threshold identification",
    "Sustained insulin target",
]
for section in sections:
    if any(s in section.splitlines()[0] for s in wanted):
        print(section[:1050])

print("\nEDIT-MACHINERY OBJECTS")
for name in sorted(list(globals())):
    if name.startswith("_"):
        continue
    if name.startswith("sc_") or re.search(r"r3|separator|checkpoint|graft|contrast", name, re.I):
        obj = globals()[name]
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            desc = f"{type(obj).__name__} {tuple(obj.shape)}"
        elif isinstance(obj, dict):
            desc = "keys=" + repr(list(obj.keys())[:14])
        elif callable(obj):
            try:
                desc = str(inspect.signature(obj))
            except Exception:
                desc = "callable"
        else:
            desc = repr(obj)[:90]
        print(name, desc)

print("\nLOCAL FUNCTION SIGNATURES")
for name in sorted(list(globals())):
    obj = globals()[name]
    if inspect.isfunction(obj) and obj.__module__ == "__main__" and not name.startswith("_"):
        try:
            print(name, inspect.signature(obj))
        except Exception:
            print(name)
```
