The marker channels track most dose changes accurately, but their factual leakage is not negligible, and norepinephrine ×3 is unstable. They are therefore not yet a validated repair. I’ll recover the architecture, record-based effect estimates, and saved implementations before choosing an edit. The hypothesis is supported only if the resulting drug signal produces the estimated effects while keeping factual NLL within the 0.005-nat limit.

```python
from pathlib import Path
import numpy as np
import inspect

arch216 = Path("/model/ARCHITECTURE.md").read_text()
notes216 = Path("/work/notes.md").read_text()
print("=== ARCHITECTURE ===")
print(arch216)
print("\n=== EARLY NOTEBOOK ===")
print(notes216[:15000])

print("\n=== NUMERIC-EDIT / IDENTIFICATION NOTEBOOK EXCERPTS ===")
paragraphs216 = notes216.split("\n\n")
matches216 = [
    p for p in paragraphs216
    if any(s in p.lower() for s in
           ["5x", "six-lag", "identification", "numeric embedding",
            "causal effect", "record-derived", "kernel estimate"])
]
for p in matches216[-22:]:
    print(p[:1800])

print("\n=== RELEVANT KERNEL OBJECTS ===")
snapshot216 = list(globals().items())
for name, obj in snapshot216:
    if name.startswith("_") or name.endswith("216"):
        continue
    if any(s in name.lower() for s in
           ["marker", "relay", "carrier", "core", "cpu", "kernel",
            "dose", "effect", "branch", "token", "attn", "prelude"]):
        if callable(obj):
            try:
                desc = str(inspect.signature(obj))
            except Exception:
                desc = type(obj).__name__
        elif hasattr(obj, "shape"):
            desc = f"{type(obj).__name__} {tuple(obj.shape)}"
        elif isinstance(obj, dict):
            desc = f"dict keys={list(obj)[:12]}"
        elif isinstance(obj, (list, tuple)):
            desc = f"{type(obj).__name__} len={len(obj)}"
        else:
            desc = type(obj).__name__
        print(name, desc)

print("\n=== ACTIVE EDITS ===", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 216\n"
            "Restart recovery: marker dose contrasts are promising, but factual "
            "carrier leakage and NE x3 instability remain unresolved. Reading "
            "architecture, identification evidence, and saved implementation "
            "objects before selecting a transmission edit. No weights changed. "
            "Counts: 216 Python runs, 20 forwards.\n")
```