The last result does not reliably identify persistence: the estimates disagree sharply across instruments. The head-only correction also failed its held-out test, so leaving the weights unchanged was appropriate. I’ll recover the architecture and earlier evidence before choosing an action-path edit; that route must improve dose sensitivity without materially worsening factual NLL.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

notes_text = Path("/work/notes.md").read_text()
print("\n=== EARLIER NOTEBOOK ===")
print(notes_text[:-8000] if len(notes_text) > 8000 else notes_text)

print("\n=== KERNEL INVENTORY ===")
for key, obj in list(globals().items()):
    if key.startswith("_") or key in {"notes_text", "obj"}:
        continue
    if inspect.isfunction(obj):
        try:
            print(key, "FUNCTION", inspect.signature(obj))
        except Exception:
            print(key, "FUNCTION")
    elif hasattr(obj, "shape"):
        print(key, type(obj).__name__, obj.shape)
    elif isinstance(obj, dict):
        print(key, "DICT", len(obj), "keys:", list(obj)[:15])
    elif isinstance(obj, (list, tuple)) and len(obj) < 30:
        print(key, type(obj).__name__, repr(obj)[:600])
print("\n=== CURRENT EDITS ===")
print(edited())
```
