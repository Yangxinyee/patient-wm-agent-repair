Run 36 did not establish a reliable reconstruction: errors remained substantial outside action bundles, and a tuple/dictionary mismatch stopped execution. No weights changed. I’ll recover the exact architecture and retained objects before designing a reversible edit. Only forward-tool validation—not reconstruction accuracy alone—can confirm that an edit preserves factual fit.

```python
from pathlib import Path
import json
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== CURRENT EDITS ===")
print(edited())

print("\n=== TENSOR NAMES AND SHAPES ===")
for name in NAMES:
    print(name, SHAPES[name])

print("\n=== VOCAB STRUCTURE / NORMALIZATION ===")
print("VOCAB type:", type(VOCAB).__name__)
if isinstance(VOCAB, dict):
    for key, value in VOCAB.items():
        if isinstance(value, dict):
            print(key, "keys:", list(value)[:40])
            if any(s in str(key).lower() for s in ("norm", "code", "event", "size")):
                print(repr(value)[:6000])
        elif isinstance(value, (list, tuple)):
            print(key, "length:", len(value), "sample:", repr(value[:6])[:800])
        else:
            print(key, repr(value)[:800])

print("\n=== RETAINED OBJECT INVENTORY ===")
_snapshot = list(globals().items())
for name, value in sorted(_snapshot):
    if name.startswith("_") or name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    typ = type(value).__name__
    if callable(value):
        if getattr(value, "__module__", "") == "__main__":
            print(name, "FUNCTION")
    elif hasattr(value, "shape"):
        extra = ""
        if hasattr(value, "columns"):
            extra = " columns=" + repr(list(value.columns)[:35])
        print(name, typ, "shape=", tuple(value.shape), extra)
    elif isinstance(value, dict):
        print(name, typ, "len=", len(value), "keys=", repr(list(value)[:18])[:650])
    elif isinstance(value, (list, tuple)):
        sample = [type(v).__name__ for v in value[:5]]
        print(name, typ, "len=", len(value), "element types=", sample)
    elif isinstance(value, (str, int, float, bool, np.number)):
        print(name, typ, repr(value)[:250])

print("\n=== EARLY NOTEBOOK: PRIOR FINDINGS AND HELPERS ===")
_notes = Path("/work/notes.md").read_text()
print(_notes[:13000])
```