The output was cut off again, so it adds no new NE evidence and no weights have changed. I’ll request only a compact kernel inventory. This will identify the existing fitting and validation functions needed to test NE while preserving the validated insulin edit.

```python
import inspect

_g167 = list(globals().items())

print("=== LOCAL FUNCTIONS ===")
for _n, _o in sorted(_g167):
    if inspect.isfunction(_o) and getattr(_o, "__module__", "") == "__main__":
        try:
            print(_n, str(inspect.signature(_o))[:180])
        except Exception:
            print(_n)

print("\n=== RECENT VARIABLES ===")
for _prefix in ["r11", "r12", "r13", "r14", "r15", "r16"]:
    print(_prefix, ", ".join(sorted(n for n,o in _g167 if n.startswith(_prefix))))

print("\n=== HEAD / NE VARIABLES ===")
print(", ".join(sorted(
    n for n,o in _g167
    if not inspect.isfunction(o) and not inspect.ismodule(o)
    and not n.startswith("r")
    and any(k in n.lower() for k in ["head", "ne_", "pair", "gate", "target", "wr_"])
)))

print("\n=== NON-RECENT DATA ARRAYS ===")
for _n, _o in sorted(_g167):
    if not _n.startswith(("_", "r")) and hasattr(_o, "shape"):
        print(_n, type(_o).__name__, _o.shape)

print("\n=== BACKUPS ===")
for _n in ["r115_strength2", "r139_candidate", "r165_insulin_backup"]:
    _o = globals().get(_n)
    print(_n, list(_o) if isinstance(_o, dict) else type(_o).__name__)
```
