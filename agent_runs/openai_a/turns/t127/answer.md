The remaining reconstruction error is too large to trust synthetic dose experiments; changing the activation did not resolve it. I’ll recover the architecture and earlier identification evidence before proceeding. An exact reconstruction—or a repair that avoids needing one—would let us test dose sensitivity safely.

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

print("\n=== RECONSTRUCTION / REPAIR HELPERS ===")
helper_names = [
    n for n, v in list(globals().items())
    if callable(v) and
    (n.startswith("rm_") or any(s in n.lower() for s in
        ("reconstruct", "graft", "posterior", "transition", "head_input", "rd_est")))
]
for n in sorted(helper_names):
    v = globals()[n]
    try:
        sig = str(inspect.signature(v))
    except Exception:
        sig = ""
    print(n, sig)

print("\n=== CURRENT EDITS ===")
print(edited())
```
