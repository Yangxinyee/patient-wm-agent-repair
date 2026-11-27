The added ungated channels all increased cached factual loss, so none is safe to adopt as screened. I’ll recover the edit helpers and earlier causal evidence before trying a smaller change. A useful candidate must improve the rollout without exhausting the factual-fit margin.

```python
from pathlib import Path
import types

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes_text = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK ===")
print(notes_text[:16000])

print("\n=== AVAILABLE FUNCTIONS ===")
print(sorted(k for k, v in list(globals().items())
             if isinstance(v, types.FunctionType) and not k.startswith("_")))

print("\n=== RELEVANT STORED OBJECTS ===")
for k, v in sorted(list(globals().items())):
    if any(s in k.lower() for s in
           ("checkpoint", "graft", "gate", "blend", "cache", "rd_", "target", "ne_")):
        if isinstance(v, dict):
            print(k, "dict", list(v.keys())[:12])
        elif hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, (int, float, str, tuple, list)):
            print(k, repr(v)[:240])
```
