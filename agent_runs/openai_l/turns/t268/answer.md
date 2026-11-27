The validated weights are now backed up, and the architecture confirms that the value-head correction also feeds into later sampled readings. I’ll recover the exact fitting functions and records-based effect evidence before increasing its strength. This should let us test a controlled change rather than guess at weight scaling.

```python
import inspect, re
from pathlib import Path

print("RECENT FUNCTIONS:")
for n in sorted(list(globals())):
    if callable(globals()[n]) and re.search(r"(21[0-9]|22[0-9]|23[0-6])$", n):
        f = globals()[n]
        try:
            sig = str(inspect.signature(f))
        except Exception:
            sig = "?"
        print(n, sig)
        if hasattr(f, "__code__"):
            print("  names:", f.__code__.co_names)

print("\nARCHITECTURE CANDIDATE:")
for k, v in arch235.items():
    if hasattr(v, "shape"):
        print(k, v.shape)
    else:
        print(k, repr(v)[:500])
print("chosen235:", {k: v for k, v in chosen235.items()
                     if k in ["train_delta", "val_delta", "tune_delta", "slopes"]})

print("\nRECENT DATA OBJECT NAMES:")
print([n for n in sorted(list(globals())) if re.search(r"(23[0-6])$", n)])

notes239 = Path("/work/notes.md").read_text()
print("\nEARLY RECORD-EFFECT EVIDENCE:")
print(notes239[:5500])
print("\nNOTES LENGTH:", len(notes239))
```