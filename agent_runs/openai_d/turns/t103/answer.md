The last run failed on a nonexistent tensor name, so it produced no evidence about the drugs or the edit. All weights remain original. I’ll recover the architecture and saved diagnostics; agreement with the GPU probe is required before trusting another local head construction.

```python
from pathlib import Path
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DEFINITIONS ===")
print(Path("/data/README.md").read_text())

print("\n=== RELEVANT TENSOR NAMES ===")
for name in NAMES:
    if any(s in name.lower() for s in
           ("gap", "query", "value", "head", "action", "drug", "dose", "norm")):
        print(name, SHAPES[name])

print("\n=== EARLY NOTEBOOK: CAUSAL ESTIMATES AND BASELINES ===")
notes = Path("/work/notes.md").read_text()
print(notes[:16000])

print("\n=== PERSISTENT VARIABLE INVENTORY ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or name in {"NAMES", "SHAPES", "VOCAB", "notes", "obj"}:
        continue
    if callable(obj):
        if getattr(obj, "__module__", "") == "__main__":
            print(name, "function")
    elif hasattr(obj, "shape"):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        keys = list(obj.keys())
        print(name, "dict", len(keys), "keys:", repr(keys[:18]))
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj))
    elif isinstance(obj, (str, int, float, bool, type(None))):
        text = repr(obj)
        if len(text) < 200:
            print(name, text)

print("\nCURRENT EDITS:", edited())
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 92: recovery after invalid tensor lookup\n"
        "Run 91 failed at core.gap.0.weight before producing a diagnostic. "
        "No edits installed. Recovering architecture, causal estimates, and caches.\n"
    )
```