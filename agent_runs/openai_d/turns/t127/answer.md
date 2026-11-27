```python
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
print("\n=== EARLY NOTEBOOK ===")
notes_text = Path("/work/notes.md").read_text()
print(notes_text[:14000])
print("\n=== CURRENT EDITS ===")
print(edited())
print("\n=== PERSISTENT VARIABLE NAMES ===")
print(sorted(k for k in globals()
             if not k.startswith("_") and k not in {"In", "Out"}))
```
