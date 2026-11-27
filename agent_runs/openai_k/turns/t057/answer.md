```python
from pathlib import Path
import inspect, dis

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECENT NOTES ===")
print(Path("/work/notes.md").read_text()[-6000:])
print("\n=== RECENT HELPERS / OBJECTS ===")
for k in sorted(list(globals())):
    if k.startswith(("r48", "r49", "r50", "r51", "rr48", "rr49", "rr50", "rr51")):
        v = globals()[k]
        if callable(v):
            print(k, inspect.signature(v))
            try:
                print(inspect.getsource(v))
            except (OSError, TypeError):
                dis.dis(v)
        elif isinstance(v, (str, dict)):
            print(k, str(v)[:2500])
        else:
            print(k, type(v).__name__, getattr(v, "shape", ""))
print("\n=== VOCAB KEYS ===")
print({k: (list(v)[:25] if isinstance(v, dict) else str(v)[:300])
       for k, v in VOCAB.items()})
print("\n=== CURRENT EDITS ===", edited())
```
