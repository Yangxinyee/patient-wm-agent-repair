The reconstruction errors are still too large to calibrate a reliable edit, and no weights are changed. I’ll reread the architecture and recover the record-derived targets. Agreement with exposed states would validate a calibration pathway; continued mismatch would rule it out.

```python
from pathlib import Path
import numpy as np

print("=== AUTHORITATIVE ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== CURRENT KERNEL OBJECTS ===")
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"In", "Out", "NAMES", "SHAPES", "VOCAB"}:
        continue
    if hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict) and len(v) <= 100:
        print(k, "dict", list(v.keys())[:30])
    elif callable(v) and getattr(v, "__module__", "") == "__main__":
        print(k, "function")

notes147 = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK: IDENTIFICATION AND TARGETS ===")
print(notes147[:13000])
print("\n=== LATER QUANTITATIVE EVIDENCE ===")
selected147 = [
    line for line in notes147.splitlines()
    if any(term in line.lower() for term in
           ["target", "instrument", "causal", "impulse", "effect estimate",
            "random", "identif", "nll:", "x2", "slope"])
]
print("\n".join(selected147)[-14000:])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Python 147 — authoritative restart audit\n"
        "- No edits active. Approximate posterior and recurrence reconstruction "
        "is not yet accurate enough for response calibration.\n"
        "- Rereading architecture and recovering record-derived targets.\n"
    )
```
