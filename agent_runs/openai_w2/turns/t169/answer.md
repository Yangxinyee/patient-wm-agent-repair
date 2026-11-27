The initial-dose constraint removes the large spurious signal while retaining most dose sensitivity. That could permit a stronger edit without harming factual fit. I’ll recover the candidate matrices and model conventions before testing it; unchanged pre-drug predictions and an NLL increase below 0.005 would support this approach.

```python
import inspect
import numpy as np

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())

print("\n=== RECORD FORMAT ===")
print(open("/data/README.md").read())

print("\n=== EARLY EVIDENCE NOTES ===")
notes148 = open("/work/notes.md").read()
print(notes148[:12000])

print("\n=== RELEVANT KERNEL OBJECTS ===")
keys148 = list(globals())
for name148 in keys148:
    if name148.startswith("_"):
        continue
    if (
        any(s in name148 for s in ["147", "134", "128", "139", "repair_head", "fallback"])
        or name148 in ["tuning_ids", "train_ids", "V128", "Wdose134", "Edecoder128"]
    ):
        obj148 = globals()[name148]
        if callable(obj148):
            try:
                desc148 = str(inspect.signature(obj148))
            except Exception:
                desc148 = type(obj148).__name__
        elif isinstance(obj148, dict):
            desc148 = "dict keys=" + repr(list(obj148.keys())[:30])
        elif hasattr(obj148, "shape"):
            desc148 = f"{type(obj148).__name__} shape={tuple(obj148.shape)}"
        else:
            desc148 = repr(obj148)[:250]
        print(name148, desc148)

print("\n=== CURRENT EDITS ===")
print(edited())
```
