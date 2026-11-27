The diagonal edit damaged factual fit, and the factorization check suggests this was not an implementation error. Only numeric gain 4 remains. I’ll recover the computation graph before choosing another pathway; the next candidate must strengthen dose responses without exceeding the factual-loss limit.

```python
from pathlib import Path

arch = Path("/model/ARCHITECTURE.md").read_text()
print("ARCHITECTURE:", len(arch), "characters")
print(arch[:16000])

print("\nPERSISTENT VARIABLE INVENTORY")
for name in sorted(list(globals())):
    if name.startswith("_") or name in {"In", "Out", "arch"}:
        continue
    obj = globals()[name]
    if callable(obj):
        continue
    shape = getattr(obj, "shape", None)
    if shape is not None:
        detail = f"shape={shape}"
    elif isinstance(obj, dict):
        detail = "keys=" + repr(list(obj.keys())[:15])
    elif isinstance(obj, (list, tuple)):
        detail = f"length={len(obj)}"
    else:
        detail = type(obj).__name__
    print(name, detail)

print("\nCURRENT OPERATIONS", ops())
```
