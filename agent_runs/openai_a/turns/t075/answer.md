The graft’s large factual NLL penalty makes it unsafe, so leaving it unapplied was correct. I’ll recover the exact architecture and retained analysis before redesigning it. A viable correction must stay nearly inactive on factual histories while responding to changed doses.

```python
from pathlib import Path
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())
print("\n=== EARLY NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:16000])

print("\n=== RETAINED FUNCTIONS ===")
for name, obj in list(globals().items()):
    if callable(obj) and getattr(obj, "__module__", None) == "__main__":
        print(name, getattr(getattr(obj, "__code__", None), "co_varnames", ())[:15])

print("\n=== RETAINED ARRAYS / TABLES / DICTIONARIES ===")
for name, obj in list(globals().items()):
    if name.startswith("_") or name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if hasattr(obj, "shape"):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), "keys:", list(obj)[:10])
    elif isinstance(obj, (list, tuple)) and len(obj) < 20:
        print(name, type(obj).__name__, str(obj)[:350])

print("\nCURRENT EDITS:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Restart after Python 65; forward 7/50\n"
            "- Gated raw-input graft was not applied: approximate factual NLL penalty +0.273.\n"
            "- Recovering exact architecture, causal evidence, and retained arrays before redesign.\n")
```