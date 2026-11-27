The stronger insulin edit increased factual NLL by 0.00266, below the 0.005 limit on these 100 stays. That supports keeping it as a candidate, not yet accepting it. I’ll recover the architecture and saved diagnostics so I can test its rollouts correctly. A larger, record-consistent insulin effect with acceptable factual fit would support the edit; insufficient improvement or excess NLL would refute it.

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
