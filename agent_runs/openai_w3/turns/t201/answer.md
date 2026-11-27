The architecture confirms that a dose signal must survive several nonlinear stages before affecting predictions. The inventory was truncated, so it did not locate the exact replay helpers. I’ll retrieve those selectively; this should show where the physical-dose contrast is being lost.

```python
import types, inspect, re, numpy as np
print("KERNEL FUNCTIONS")
for name in sorted(list(globals())):
    obj = globals()[name]
    if isinstance(obj, types.FunctionType) and getattr(obj, "__module__", "") == "__main__":
        try:
            print(name, str(inspect.signature(obj))[:160])
        except Exception:
            print(name)

print("\nRECENT NAMES ONLY")
names181 = [n for n in sorted(list(globals()))
            if re.search(r"(154|169|171|174|175|176|177|178|179)$", n)]
print(" ".join(names181))

print("\nCANDIDATE STRUCTURES")
for name in ["post154", "zin154", "zb154", "effect171", "zero169",
             "reference", "ref", "scenarios", "cases", "train", "df"]:
    if name not in globals():
        continue
    obj = globals()[name]
    if isinstance(obj, dict):
        print(name, [(k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                     for k, v in obj.items()])
    elif hasattr(obj, "shape"):
        print(name, type(obj).__name__, obj.shape)
    else:
        print(name, type(obj).__name__, str(obj)[:300])

text181 = open("/work/notes.md").read()
print("\nEARLY RECORD EVIDENCE")
for paragraph in text181.split("\n\n"):
    if re.search(r"(instrument|randomiz|identif|causal|kernel|confound)", paragraph, re.I):
        print(paragraph[:1100])
        if "printed181" not in globals():
            printed181 = 0
        printed181 += 1
        if printed181 >= 7:
            break

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 181; forwards 10. Architecture confirms dose transport passes nonlinear posterior/prelude/recurrence. Run 180 inventory truncated; retrieving selective helper signatures and record-identification notes. No weights changed.\n")
```
