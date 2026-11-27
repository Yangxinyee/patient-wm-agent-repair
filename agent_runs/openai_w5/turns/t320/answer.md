The latest edit increased insulin’s doubling response, but factual NLL still fails the limit and halving remains much weaker. I’ll recover the record-derived targets and edit structure before changing anything. Acceptance requires both target agreement and an NLL increase below 0.005.

```python
from pathlib import Path
import inspect, re

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA README ===")
print(Path("/data/README.md").read_text())

notes = Path("/work/notes.md").read_text()
print("=== EARLY NOTEBOOK / IDENTIFICATION ===")
print(notes[:22000])
print("=== LATER TARGET / CALIBRATION EVIDENCE ===")
lines = notes.splitlines()
hits = [i for i, s in enumerate(lines)
        if re.search(r"target|causal|identified|instrument|estimate|true effect|record-derived", s, re.I)]
shown = set()
for i in hits:
    if sum(len(s) + 1 for s in lines[:i]) < 22000:
        continue
    for j in range(max(0, i-1), min(len(lines), i+3)):
        if j not in shown:
            print(lines[j])
            shown.add(j)

print("=== KERNEL OBJECTS ===")
for name, obj in list(globals().items()):
    if name.startswith("_") or name in {"notes", "lines", "hits", "shown", "obj"}:
        continue
    if inspect.isfunction(obj):
        try:
            print("FUNCTION", name, inspect.signature(obj))
        except Exception:
            print("FUNCTION", name)
    elif re.search(r"base|center|trial|panel|state|target|coef|fit|rate|pair|strength|cache|keep", name, re.I):
        desc = getattr(obj, "shape", None)
        if isinstance(obj, dict):
            desc = ("dict", len(obj), list(obj)[:12])
        print(name, type(obj).__name__, desc)
```
