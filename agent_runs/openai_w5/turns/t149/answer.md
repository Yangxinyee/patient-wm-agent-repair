The memory channel carries dose information, but at a much smaller scale than intended; the output also has a baseline offset. This verifies transport, not a causal repair. I’ll review the architecture and earlier record-based estimates before tracing the signal to the reading head. The next checks must show reliable dose recovery without exhausting the factual-loss budget.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA DESCRIPTION ===")
print(Path("/data/README.md").read_text())

notes130 = Path("/work/notes.md").read_text()
print("=== EARLIER NOTEBOOK ===")
print(notes130[:18000])
print("Notebook characters:", len(notes130))

print("=== PERSISTENT HELPERS ===")
for name130, obj130 in sorted(list(globals().items())):
    if inspect.isfunction(obj130) and not name130.startswith("_"):
        try:
            sig130 = str(inspect.signature(obj130))
        except Exception:
            sig130 = "(unknown)"
        print(name130, sig130)

print("=== RECENT OBJECTS ===")
for name130, obj130 in sorted(list(globals().items())):
    if name130.endswith(("105", "117", "118", "123", "125", "127", "128", "129")):
        if isinstance(obj130, dict):
            print(name130, "dict keys:", list(obj130.keys())[:35])
        elif hasattr(obj130, "shape"):
            print(name130, type(obj130).__name__, tuple(obj130.shape))
        elif isinstance(obj130, (str, int, float, bool, list, tuple)):
            text130 = repr(obj130)
            print(name130, text130[:250])

print("=== CURRENT EDIT COUNT ===", len(edited()))
```
