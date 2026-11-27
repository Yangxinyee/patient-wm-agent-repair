The notebook supports substantial drug effects estimated from the records, but the last output was truncated before the repair details appeared. I’ll retrieve only the relevant kernels and candidate interfaces. Comparing those kernels with the model’s impulse responses should tell us whether the edit has the wrong persistence or an incorrect initial response.

```python
import re
print("=== SELECTED NOTEBOOK ENTRIES ===")
for section in Path("/work/notes.md").read_text().split("\n## "):
    heading = section.splitlines()[0] if section else ""
    m = re.match(r"Run(?:s)?\s+(\d+)", heading)
    if m and int(m.group(1)) in {137, 184, 186, 190, 193, 194, 195}:
        print("##", section[:850])

print("\n=== LATE CANDIDATE INTERFACES ===")
for name, obj in list(globals().items()):
    if re.search(r"(193|194|195|206|209|210)$", name):
        if inspect.isfunction(obj):
            print(name, str(inspect.signature(obj)),
                  "locals", obj.__code__.co_varnames,
                  "globals", obj.__code__.co_names)
        elif isinstance(obj, dict):
            print(name, "dict", list(obj)[:12])
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, obj.shape)
        elif isinstance(obj, (int, float, str, list, tuple)):
            print(name, repr(obj)[:140])

print("\n=== KERNEL/RESPONSE OBJECT NAMES ===")
for name, obj in list(globals().items()):
    if any(s in name.lower() for s in ["kernel", "readout_estimates", "quadratic137"]):
        if isinstance(obj, dict):
            print(name, "dict", list(obj)[:10])
            for k, v in list(obj.items())[:2]:
                print(" ", k, repr(v)[:450])
        elif hasattr(obj, "shape"):
            print(name, obj.shape)
        elif inspect.isfunction(obj):
            print(name, inspect.signature(obj))

print("\n=== ORIGINAL MODEL IMPULSE ESTIMATES ===")
for drug, tab in impulse_tables_repair.items():
    print(drug, tab.to_dict("list"))
print("\n=== EDITED IMPULSE SUMMARIES ===")
for key, val in impulse_tests166.items():
    print(key, repr(val)[:500])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 221\nRun-220 output was truncated before late construction details. "
            "Retrieved targeted kernel and candidate metadata with bounded output. "
            "No saved weights changed; checking initial response and persistence against record-derived kernels.\n")
```
