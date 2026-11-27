The last run failed at normalization lookup, before testing or changing weights. The keys suggest an event-and-field pair is needed. I’ll verify that format and recover the exact prediction path. Agreement with known predictions would justify testing a head correction; disagreement would mean the reconstruction still needs repair.

```python
from pathlib import Path
import types

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

def r174_summary(x, depth=0):
    if hasattr(x, "shape"):
        return f"{type(x).__name__} shape={tuple(x.shape)}"
    if isinstance(x, dict):
        keys = list(x)
        if depth >= 1:
            return f"dict keys={keys[:12]}"
        return {str(k): r174_summary(x[k], depth+1) for k in keys[:12]}
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__} len={len(x)}"
    if isinstance(x, (int, float, str, bool, type(None))):
        return str(x)[:180]
    return type(x).__name__

print("=== VOCAB STRUCTURE ===")
print(r174_summary(VOCAB))
for k, v in VOCAB.items():
    if "norm" in str(k).lower():
        print("normalization entry:", k, r174_summary(v))
        if isinstance(v, dict):
            for kk, vv in v.items():
                if any(s in str(kk) for s in ("glucose_finger", "map_arterial", "insulin")):
                    print(kk, r174_summary(vv))

print("=== RECENT INTERFACES ===")
r174_snapshot = list(globals().items())
for name, obj in r174_snapshot:
    relevant = any(s in name.lower() for s in
                   ("head", "readout", "local", "decode", "candidate139",
                    "candidate158", "normal", "context", "contrast", "paired"))
    if not relevant:
        continue
    if isinstance(obj, types.FunctionType):
        c = obj.__code__
        print(name, "FUNCTION", c.co_varnames[:c.co_argcount],
              "uses", c.co_names[:24])
    elif not name.startswith("__"):
        print(name, r174_summary(obj))

r174_notes = Path("/work/notes.md").read_text()
print("=== EARLY IDENTIFICATION NOTES ===")
print(r174_notes[:8500])
for tag in ("Run 139", "Run 171", "Run 172", "Run 173"):
    pos = r174_notes.rfind(tag)
    if pos >= 0:
        print("=== NOTE", tag, "===")
        print(r174_notes[max(0,pos-3):pos+1800])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 174: restart and normalization recovery\n"
            "- Run 173 stopped at a normalization lookup; no new edit was evaluated.\n"
            "- Recovering architecture, physical units, retained identification evidence, "
            "and exact readout interfaces before the next candidate.\n"
            "- Python runs: 174; forward passes: 10. Saved edits unchanged.\n")
```
