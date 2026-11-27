The reconstructed heads still disagree with the forward tool, especially for glucose. That is too much uncertainty for a safe weight edit. I’ll recover the architecture and retained analysis objects next; agreement after correcting state timing and normalization would validate a local editing method.

```python
from pathlib import Path
import json, inspect
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DESCRIPTION ===")
print(Path("/data/README.md").read_text())
print("\n=== FULL NOTEBOOK ===")
print(Path("/work/notes.md").read_text())

print("\n=== TENSOR INVENTORY ===")
for name in NAMES:
    print(name, SHAPES[name])

print("\n=== RETAINED OBJECTS ===")
for name, obj in list(globals().items()):
    if name.startswith("_") or name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(obj, np.ndarray):
        print(name, "ndarray", obj.shape, obj.dtype)
    elif hasattr(obj, "columns") and hasattr(obj, "shape"):
        print(name, type(obj).__name__, obj.shape, list(obj.columns)[:35])
    elif isinstance(obj, dict):
        print(name, "dict keys:", list(obj)[:30])
    elif isinstance(obj, (list, tuple)) and len(obj) < 150:
        print(name, type(obj).__name__, str(obj)[:800])
    elif inspect.isfunction(obj):
        print(name, "function", inspect.signature(obj))
    elif isinstance(obj, (str, int, float, bool)):
        print(name, repr(obj)[:300])

print("\n=== VOCAB ===")
print(json.dumps(VOCAB, indent=2)[:24000])
print("\n=== CURRENT EDITS ===", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Restart — Python 10; forward 4\n"
        "- No weights edited. Local head reconstruction is not yet exact; "
        "recovering architecture and retained objects before designing a reversible edit.\n"
    )
```